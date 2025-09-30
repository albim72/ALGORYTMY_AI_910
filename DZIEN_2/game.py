# -*- coding: utf-8 -*-
"""
NarrativeMap: interaktywna mapa narracyjna do gier fabularnych.

Wymaga:
    pip install networkx
"""

from __future__ import annotations
import itertools
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import networkx as nx


@dataclass(frozen=True)
class Decision:
    """Reprezentuje pojedynczą gałąź decyzji w drzewie."""
    label: str
    risk: int
    reward: int


class NarrativeMap:
    """
    Klasa budująca narracyjne drzewo decyzyjne oparte o motywy, miejsca i postaci.

    Drzewo jest przechowywane w ``networkx.DiGraph``. Węzły zawierają
    automatycznie generowane opisy zdarzeń, a krawędzie – etykiety wyborów
    (np. „zaufaj”, „zdrada”) oraz wartości ryzyka/nagrody (1–10).

    Parametry
    ----------
    themes : List[str]
        Lista motywów przewodnich (np. ["odkupienie", "tajemnica"]).
    locations : List[str]
        Lista lokacji (np. ["opuszczony zamek", "mglista dolina"]).
    characters : List[str]
        Lista postaci (np. ["Wędrowiec", "Cień"]).
    depth : int, optional
        Głębokość drzewa decyzyjnego (liczba poziomów wyborów), domyślnie 3.
    branching : int, optional
        Liczba gałęzi z każdego węzła (2 = binarne decyzje), domyślnie 2.
    seed : Optional[int], optional
        Ziarno generatora losowego dla powtarzalności, domyślnie None.

    Atrybuty
    --------
    graph : nx.DiGraph
        Skierowany graf przechowujący drzewo narracyjne.
    root : Any
        Identyfikator węzła korzenia (zwykle 0).
    decisions_catalog : List[Tuple[str, str]]
        Para przeciwstawnych etykiet wyboru (np. ("zaufaj", "zdrada")).
    """

    def __init__(
        self,
        themes: List[str],
        locations: List[str],
        characters: List[str],
        depth: int = 3,
        branching: int = 2,
        seed: Optional[int] = 42,
    ) -> None:
        if branching < 2:
            raise ValueError("Parametr 'branching' musi być >= 2.")

        self.themes = themes
        self.locations = locations
        self.characters = characters
        self.depth = depth
        self.branching = branching
        self.root = 0
        self.graph: nx.DiGraph = nx.DiGraph()
        self._rng = random.Random(seed)

        # Katalog par decyzji (etykiety). Będą rotacyjnie używane na poziomach.
        self.decisions_catalog: List[Tuple[str, str]] = [
            ("zaufaj", "zdrada"),
            ("pośpiech", "rozwaga"),
            ("poświęcenie", "egoizm"),
            ("prawda", "zatajenie"),
            ("rytuał", "profanacja"),
            ("sojusz", "samotność"),
        ]

        self._build_tree()

    # --------------------- API PUBLICZNE ---------------------

    def get_path_summary(self, path: List[Any]) -> str:
        """
        Zwraca opis przebiegu zadanej ścieżki wraz z podsumowaniem ryzyka i nagrody.

        Parametry
        ----------
        path : List[Any]
            Lista identyfikatorów węzłów tworzących ścieżkę od korzenia do liścia
            (np. wynik ``nx.shortest_path`` lub kolejne ``child_id`` zwracane przez
            użytkownika w trakcie eksploracji). Ścieżka musi zaczynać się w korzeniu.

        Zwraca
        -------
        str
            Tekstowa synteza: opisy zdarzeń na kolejnych węzłach, podjęte wybory
            na krawędziach oraz łączna suma ryzyka i nagrody.

        Podnosi
        -------
        ValueError
            Jeśli ścieżka nie zaczyna się w korzeniu lub nie jest spójna w grafie.
        """
        if not path or path[0] != self.root:
            raise ValueError("Ścieżka musi zaczynać się w korzeniu (node 0).")

        total_risk = 0
        total_reward = 0
        lines: List[str] = []

        # Opis pierwszego węzła
        lines.append(f"[Start] {self.graph.nodes[path[0]]['description']}")

        # Przejścia po krawędziach
        for u, v in zip(path, path[1:]):
            if not self.graph.has_edge(u, v):
                raise ValueError(f"Niepoprawna ścieżka: brak krawędzi {u} -> {v}.")
            edge = self.graph.edges[u, v]
            label = edge["label"]
            risk = edge["risk"]
            reward = edge["reward"]
            total_risk += risk
            total_reward += reward

            lines.append(f" → [Wybór: {label}] (Ryzyko {risk}, Nagroda {reward})")
            lines.append(f"   {self.graph.nodes[v]['description']}")

        lines.append(
            f"\n[Podsumowanie] Łączne Ryzyko: {total_risk} | Łączna Nagroda: {total_reward}"
        )
        return "\n".join(lines)

    def get_leaves(self) -> List[Any]:
        """
        Zwraca listę identyfikatorów liści w drzewie.

        Zwraca
        -------
        List[Any]
            Lista węzłów, które nie posiadają dzieci (out-degree == 0).
        """
        return [n for n in self.graph.nodes if self.graph.out_degree(n) == 0]

    def get_root_to_leaf_paths(self) -> List[List[Any]]:
        """
        Zwraca wszystkie ścieżki od korzenia do liści.

        Zwraca
        -------
        List[List[Any]]
            Lista ścieżek; każda ścieżka to lista identyfikatorów węzłów.
        """
        leaves = self.get_leaves()
        paths: List[List[Any]] = []
        for leaf in leaves:
            for path in nx.all_simple_paths(self.graph, source=self.root, target=leaf):
                paths.append(path)
        return paths

    # --------------------- BUDOWA DRZEWA ---------------------

    def _build_tree(self) -> None:
        """
        Buduje drzewo decyzyjne o zadanej głębokości i rozgałęzieniu.

        Tworzy korzeń (node 0) oraz rekurencyjnie dodaje dzieci z krawędziami
        zawierającymi etykiety wyborów oraz wartości ryzyka/nagrody (1–10).
        """
        self.graph.add_node(self.root, description=self._generate_event_description(self.root))
        current_level = [self.root]
        node_id_counter = 1

        for level in range(self.depth):
            next_level = []
            label_pair = self.decisions_catalog[level % len(self.decisions_catalog)]

            # Jeśli branching > 2, rozszerzamy etykiety rotacyjnie
            labels = self._expand_labels(label_pair, self.branching)

            for parent in current_level:
                for i in range(self.branching):
                    child = node_id_counter
                    node_id_counter += 1

                    self.graph.add_node(child, description=self._generate_event_description(child))
                    decision = self._sample_decision(labels[i])
                    self.graph.add_edge(
                        parent,
                        child,
                        label=decision.label,
                        risk=decision.risk,
                        reward=decision.reward,
                    )
                    next_level.append(child)

            current_level = next_level

    def _expand_labels(self, pair: Tuple[str, str], branching: int) -> List[str]:
        """
        Rozszerza parę etykiet do listy długości ``branching``, naprzemiennie.

        Parametry
        ----------
        pair : Tuple[str, str]
            Dwie przeciwstawne etykiety (np. ("zaufaj", "zdrada")).
        branching : int
            Docelowa liczba etykiet.

        Zwraca
        -------
        List[str]
            Lista etykiet długości ``branching``.
        """
        base = list(pair)
        out = []
        idx = 0
        for _ in range(branching):
            out.append(base[idx % len(base)])
            idx += 1
        return out

    # --------------------- GENEROWANIE TREŚCI ---------------------

    def _generate_event_description(self, node_id: Any) -> str:
        """
        Generuje krótki (2–3 zdania) opis zdarzenia dla danego węzła.

        W tej implementacji opis jest syntetyzowany lokalnie z motywów, miejsc
        i postaci. W produkcji tutaj można podpiąć model językowy.

        Parametry
        ----------
        node_id : Any
            Identyfikator węzła, dla którego generujemy opis.

        Zwraca
        -------
        str
            Dwuzdaniowy/trzyzdaniowy opis fabularny.

        TODO
        ----
        call GPT here
        """
        theme = self._rng.choice(self.themes) if self.themes else "tajemnica"
        loc = self._rng.choice(self.locations) if self.locations else "nieznane miejsce"
        chars = ", ".join(self._rng.sample(self.characters, k=min(2, max(1, len(self.characters))))) if self.characters else "Nieznajomy"
        clue = self._rng.choice(
            [
                "stary symbol",
                "zakazany szept",
                "połamany medalion",
                "mapa bez legendy",
                "ślad popiołu",
                "echo kroków",
            ]
        )
        twist = self._rng.choice(
            [
                "Pomoc nadchodzi z niespodziewanej strony.",
                "Cena prawdy okazuje się wyższa niż sądzono.",
                "Cień przeszłości domaga się odpowiedzi.",
                "Odkupienie wymaga intymnego poświęcenia.",
                "Zaufanie staje się narzędziem zdrady.",
            ]
        )
        # 2–3 zdania
        sentences = [
            f"W {loc} motyw {theme} splata losy: {chars}.",
            f"Na miejscu odnajdujecie {clue}, który zmienia rozkład sił.",
            twist,
        ]
        # Losowo 2 lub 3 zdania, ale preferuj 2–3
        n = self._rng.choice([2, 3, 3])
        return " ".join(sentences[:n])

    def _sample_decision(self, label: str) -> Decision:
        """
        Losuje wartości ryzyka i nagrody (1–10) dla danej etykiety wyboru.

        Parametry
        ----------
        label : str
            Etykieta wyboru (np. „zaufaj”, „zdrada”).

        Zwraca
        -------
        Decision
            Struktura zawierająca etykietę oraz wylosowane ryzyko/nagrodę.

        Notatki
        -------
        Rozkład może być lekko stronniczy: np. „poświęcenie” ma tendencję
        do wyższej nagrody; „pośpiech” – wyższego ryzyka.
        """
        # Biasy tematyczne (subtelne, ale odczuwalne)
        bias: Dict[str, Tuple[int, int]] = {
            "zaufaj": (0, +1),
            "zdrada": (+1, 0),
            "pośpiech": (+2, 0),
            "rozwaga": (-1, 0),
            "poświęcenie": (0, +2),
            "egoizm": (0, +1),
            "prawda": (0, +1),
            "zatajenie": (+1, 0),
            "rytuał": (+1, +1),
            "profanacja": (+2, 0),
            "sojusz": (0, +1),
            "samotność": (+1, 0),
        }
        br, bw = bias.get(label, (0, 0))

        def clamp01(x: int) -> int:
            return max(1, min(10, x))

        risk = clamp01(self._rng.randint(1, 10) + br)
        reward = clamp01(self._rng.randint(1, 10) + bw)
        return Decision(label=label, risk=risk, reward=reward)


# --------------------- PRZYKŁADOWE UŻYCIE ---------------------
if __name__ == "__main__":
    themes = ["odkupienie", "tajemnica"]
    locations = ["opuszczony zamek", "mglista dolina"]
    characters = ["Wędrowiec", "Cień"]

    nm = NarrativeMap(themes, locations, characters, depth=3, branching=2, seed=7)

    # Wypisz wszystkie ścieżki root->leaf i ich podsumowania
    paths = nm.get_root_to_leaf_paths()
    print(f"Liczba ścieżek: {len(paths)}\n")

    # Pokaż 1–2 przykładowe ścieżki
    for i, p in enumerate(paths[:2], start=1):
        print(f"--- Ścieżka {i} (węzły): {p}")
        print(nm.get_path_summary(p))
        print()
