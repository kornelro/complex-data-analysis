Badania klasyfikacji hierarchicznej zostały przeprowadzone na zbiorze imclef07a, w którym każda próbka przypisana jest do liścia w drzewie hierarchii.
Jako bazowe modele wykorzystane zostały drzewa decyzyjne z ograniczeniem maksymalnej głębokości do 5 poziomów lub bez takiego ograniczenia.
Zbadane zostały podejścia LCPN, LCN, LCL, FLAT, które zostały zaimplementowane, a dodatkowo wyniki LCPN, LCN zostały porównane z w wynikami zwracanymi przez implementacje z biblioteki sklearn_hierarchical_classification.
Do oceny podejść wykorzystane zostały hierarchiczne i zwykłe F1Score.

Po przeprowadzeniu badań można powiedzieć, że:
- Klasyfikacja hierarchiczna nie jest potrzebna gdy problem jest wystarczająco prosty dla bazowego klasyfikatora w podejściu FLAT - gdy nie przycinamy drzewa podejście FLAT daje najlepsze wyniki, jeśli jednak osłabiamy drzewo przycięciem to klasyfikacja hierarchiczna znacząco te wyniki poprawia.
- Z podejść hierarchicznych najlepiej sprawdza się podejście LCN, a najgorzej LCL (wyniki gorsze od podejścia FLAT). Podejście LCPN jest nieco gorsze od LCN, ale jest zbudowane z mniejszej liczby modeli przez co działa szybciej.
- Własne implementacje dają zbliżone wyniki i działają szybciej niż implementacje z biblioteki, jednak prawdopodobnie wynika to z tego, że zostały one dostosowane do badanego problemu - istnieje w nich założenie, że każda próbka w zbiorze ma swoją klasę w liściu drzewa hierarchii. Implementacje z biblioteki powinny obsługiwać również inne przypadki.