W zadaniu do detekcji *concept drift* wykorzystano metodę, która porównuje napływające próbki ze zbiorem referencyjnym i za pomocą testu *Kolmogorov–Smirnov* ocenia czy nowe próbki pochodzą z tego samego rozkładu (wykorzystano implementację z biblioteki *alibi-detect*). Zbiór referencyjny o wielkości N jest przygotowywany poprzez wczytanie pierwszych N napływających próbek, a następnie może pozostać bez zmian lub może być stale akutalizowany do N ostatnich wczytanych próbek. Napływające nowe próbki są zbierane w zbiory testowe o wielkości L i na podstawie tych zbiorów oceniano wystąpienie *concept drift*. Stosowano wielkość zbioru referencyjnego równą 1000 próbek, wielkość zbioru testowego równą 100, a dla testu statystycznego przyjęto p_value=0.005.

Do wczytywania danych jako danych streamowych (wczytywanie kolejno pojedynczych próbek zamiast całego pliku) wykorzystano wbudowany w Pythonie *csv reader*.

Do estymacji rozkłady przeyjęto, że wszystkie dane pochodzą z rozkładu normalnego, a następnie podczas odczytywania kolejnych próbek zapisywano okno K próbek i po każdym zapełnieniu okna empirycznie wyznaczano parametry rozkładu. Stosowano okno wielkości 1000 próbek.

Wizualizacje wyników przedstawione w *results_analysis.ipynb* pokazują, że:
- wykorzystana metoda skutecznie oznacza zmiany rozkładów,
- aktualizowanie zbioru referencyjnego, powoduje, że jeżeli próbki całkowicie zmieniają swój rozkład, to detektor dostosowuje się do nowego rozkładu,
- to czy stosowanie aktualizacji zbioru referencyjnego jest poprawne z pewnością zależy od konkretnego problemu,
- rozkłady wyznaczane w kolejnych oknach dobrze odwzorowują zmiany widoczne na serii czasowej,
- w pierwszej serii możemy zauważyć nagłe zmiany rozkładów,
- w drugiej serii mamy doczynienia ze zmianą narastającą,
- w trzeciej serii nie dochodzi do zmiany rozkładu.
