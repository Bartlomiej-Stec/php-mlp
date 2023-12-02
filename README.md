# Trójwarstwowa sieć neuronowa typu MLP z przyspieszeniem metodą momentum i adaptacyjnym współczynnikiem uczenia

Biblioteka umożliwia wykorzystanie trójwarstwowej sieci neuronowej w PHP. W implementacji wykorzystano w pierwszej oraz drugiej warstwie ukrytej tangensoidalną funkcję aktywacji oraz liniową na wyjściu. Bilbioteka umożliwia ustawienie wszystkich stosowanych parametrów takich jak liczba neuronów w poszczególnych warstwach, learning rate, czy błąd docelowy. Parametry wyuczonej sieci można w prosty sposób zapisać na dysku, aby później użyć je do przewidywania wcześniej nieznanych danych bez przechodzenia ponownego procesu uczenia. Możliwe jest również poszukiwanie optymalnych parametrów, dla których sieć daje najlepsze wyniki.
## Instalacja

Do pliku composer.json należy dodać do sekcji `repositories`

```bash
  {
    "type": "vcs",
    "url": "git://github.com/Bartlomiej-Stec/php-mlp.git"
  }
```
Następnie w tym samym pliku w sekcji `require`

```bash
    "vendor/my-private-repo": "dev-master"
```
Po dodaniu powyższych linii należy użyć komendy `composer install`

Aby użyć zainstalowanej biblioteki należy dołączyć odpowiednie pliki w kodzie PHP:
```php
<?php
    require 'vendor/autoload.php';
    use NeuralNetwork\MLP;
    use NeuralNetwork\Optimizer;
```

## Dokumentacja

Biblioteka posiada 2 główne klasy: `MLP` oraz `Optimizer`. Ta druga służy jedynie do znajdywania optymalnych parametrów sieci. Aby rozpocząć korzystanie z sieci należy stworzyć obiekt MLP:
```php
  $mlpnet = new MLP(count($x[0]), $max_epoch, $K1, $K2, $K3, $lr, $err_goal, $mc, $ksi_inc, $ksi_dec, $er);
```
| Paramter | Typ     | Opis                |
| :-------- | :------- | :------------------------- |
| `l_input` | `int` | **Wymagane**. Liczba cech  |
| `max_epoch` | `int` | Maksymalna liczba iteracji  |
| `K1` | `int` | Liczba neuronów w pierwszej warstwie ukrytej  |
| `K2` | `int` | Liczba neuronów w drugiej warstwie ukrytej  |
| `K3` | `int` | Liczba neuronów na wyjściu (zazwyczaj tyle samo, co liczba klas)  |
| `lr` | `float` | Learning rate (współczynnik uczenia sieci) |
| `err_goal` | `float` | Błąd docelowy (po jego osiągnięciu uczenie jest przerywane) |
| `mc` | `float` | Momentum (przyjmuje wartości z zakresu od 0 do 1) |
| `ksi_inc` | `float` | Współczynnik zwiększania learing rate (powinien być większy niż 1) |
| `ksi_dec` | `float` | Współczynnik zmniejszania learning rate (powinien być mniejszy niż 1) |
| `er` | `float` | Dopuszczalna krotność przyrostu błędu (wartość dla której lr nie zostanie zmieniona) |

Aby rozpocząć uczenie sieci należy wywołać metodę `train` podając jako parametr 2D array z danymi wejściowymi oraz 2D array z oczekiwanymi rezultatami.

```php
  $x = [
    [1, 2, 2, 2],
    [5, 5, 5, 6]
  ]; //4 cechy
  $y = [
    [1],
    [2]
  ]; //1 klasa
  $mlpnet->train($x, $y);
```
Po zakończeniu procesu uczenia można skorzystać z metody `predict`, która zwraca przewidywane rezultaty dla podanego wejścia:

```php
  $x = [
    [2, 2, 2, 2]
  ];
  $mlpnet->predict($x);
```
Metoda zwraca array 2D z wartościami typu float. Nie są one równe. Przynależność do poszczególnej klasy możliwa jest do określenia stosując zaokrąglenia. Im wartość jest bliższa liczbie całkowitej, tym sieć ,,jest bardziej pewna", że wyznaczona klasa jest właściwa. Jest to szczególnie przydatne w przypadku gdy klasy są kolejno ponumerowane i służą do wyznaczania oceny czegoś w określonej skali. W przypadku błędnej klasyfikacji istnieje większe prawdopodobieństwo, że błędnie wyznaczona klasa jest jedną z sąsiednich, co w pewnych przypadkach mimo wszystko będzie wystarczające.

Za pomocą metody `roundOutput` można zaokrąglić rezultaty do pełnych liczb, odpowiadających klasom.
```php
  $result = $mlpnet->predict($x); //Output [[3.6]]
  $mlpnet->roundOutput($result);
  print_r($result); //Output [[4]]
```

W procesie uczenia można zastosować walidację krzyżową korzystając z metody `crossValidationTrain`:
| Paramter | Typ     | Opis                |
| :-------- | :------- | :------------------------- |
| `x` | `array` | **Wymagane**. Dane treningowe  |
| `y` | `array` | **Wymagane**. Przewidywane wyjście danych treningowych  |
| `CVN` | `int` | Ilość podziałów danych w walidacji krzyżowej  |
| `print_results` | `bool` | Przy ustawionej wartości na true poszczególne wyniki są wyświetlane  |

W wyniku metoda zwraca średnie PK (poprawność klasyfikacji) w procentach.

Parametry nauczonego modelu sieci takie jak np. wagi, biasy można zapisać na dysku stosując metodę `saveModel`. Jako argument można podać nazwę pliku. Zapisane modele znajdują się w folderze `models`. Można również skorzystać z metody `getModel`, aby pobrać parametry i zapisać je w inny sposób.

Aby załadować parametry modelu z dysku można skorzystać z metody `loadSavedModel` podając nazwę zapisanego wcześniej modelu lub podać parametry modelu do metody `loadModel`. W obu przypadkach zainicjowane losowymi wartościami wagi oraz biasy zostaną zastąpione podanymi wartościami.

```php
  $result = $mlpnet->train($x, $y);
  $mlpnet->saveModel('example_classification');
  $mlpnet->loadSavedModel('example_classification');
```

Aby znaleźć optymalne parametry sieci można użyć klasy `Optimizer`. Składa się ona z metod do wyznaczania optymalnych wartości `K1, K2` oraz `lr, err_goal, max_epoch`. Nie jest jednak możliwe znalezienie idealnych parametrów z gwarancją, że na pewno są one najlepsze. Sprawdzenie wszystkich możliwych kombinacji nawet z ustawionym limitem może zająć wiele czasu, dlatego sprawdzane są K1, K2 przy założeniu że pozostałe parametry są optymalne, oraz lr, err_goal, max_epoch przy takim samym założeniu. Dzięki temu w sensownym czasie zwykle możliwe jest znalezienie wystarczająco dobrych parametrów. Metody korzystają z walidacji krzyżowej wykonywanej wielokrotnie (w zależności od podanego parametru) i na koniec wyliczają medianę z otrzymanych rezultatów. Dla każdego eksperymentu wyświetlany jest jego numer oraz najlepsze do tej pory znalezione parametry. W przypadku metody `findOptimalK` należy podać dane treningowe, przewidywane wyjście, maksymalną wartość `K1`, `K2` oraz krok dla każdej z nich. Można również podać pozostałe parametry takie same jak w przypadku obiektu `MLP`. Z kolei metoda `findOptimalLrEpochErr` przyjmuje jako parametry dane treningowe, przewidywane wyjście, 2D array wartości, które mają zostać sprawdzone dla `max_epoch`, `lr` oraz `err_goal`. Są to jednak parametry opcjonalne, które na wypadek ich nieuzupełnienia zostaną zastąpione wartościami domyślnymi. 


