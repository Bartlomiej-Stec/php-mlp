<?php

namespace NeuralNetwork;

class NNet
{
    protected function randX($a, $b)
    {
        $P = array();
        for ($i = 0; $i < $a; $i++) {
            $row = array();
            for ($j = 0; $j < $b; $j++) {
                $row[] = rand() / getrandmax();
            }
            $P[] = $row;
        }
        return $P;
    }

    protected function randn($a, $b) //tworzy tablice losowych wartosci z przedzialu od -1 do 1
    {
        $w = $this->randX($a, $b);
        for ($i = 0; $i < $a; $i++) {
            for ($j = 0; $j < $b; $j++) {
                $w[$i][$j] = $w[$i][$j] * 2.0 - 1.0;
            }
        }
        return $w;
    }

    protected function nwtan($s, $p)
    {
        //wielkosc wag powinna byc skalowana w odniesieniu do liczby neuronow s i liczby wejsc p
        $magw = 0.7 * pow($s, 1.0 / $p); //0.7 stala heurystyczna, pow(s, 1/p) - srednia geometryczna
        $w = Math::scalarMultiply($magw, $this->normRows($this->randn($s, $p))); //wagi w normalizowane losowe wartosci tablicy 2d skalowane przez magw
        $b = Math::scalarMultiply($magw, $this->randn($s, 1)); //wektor b wielkosci liczby neuronow w warstwie
        $rng = Math::staticNumberArray(2, 1, $p); //tworzy tablice wypeniona 2
        $mid = Math::staticNumberArray(0, $p, 1); //tworzy tablice wypelniona 0
        //skaluje wagi 2 i dzieli przez dot. Operacja ma na celu wysrodkowac wartosci wokol 0
        $w = Math::arrayDivide(Math::scalarMultiply(2.0, $w), Math::dot(Math::staticNumberArray(1, $s, 1), $rng));
        $b = Math::subtract($b, Math::dot($w, $mid)); //Operacja ma na celu zapewnic ze wartosci sa symetryczne wokol 0
        return array($w, $b);
    }

    protected function normRows($a) //normalizuje dane (normalizacja euklidesa)
    {
        //kazda wartosc dzielona jest przez pierwiastek sumy kwadratow
        //||v|| = sqrt(sum(v[i]^2)) gdzie i=1 do n
        $P = $a;
        list($rows, $columns) = Math::getShape($a);
        for ($x = 0; $x < $rows; $x++) {
            $sumSq = 0;
            for ($y = 0; $y < $columns; $y++) {
                $v = $P[$x][$y];
                $sumSq += $v ** 2.0;
            }
            $len = sqrt($sumSq);
            for ($z = 0; $z < $columns; $z++) {
                $P[$x][$z] = $P[$x][$z] / $len;
            }
        }
        return $P;
    }

    protected function rands($a, $b) //tworzy tablice 2d losowych wartosci od -1 do 1
    {
        $w = $this->randX($a, $b);
        for ($i = 0; $i < $a; $i++) {
            for ($j = 0; $j < $b; $j++) {
                $w[$i][$j] = $w[$i][$j] * 2.0 - 1.0;
            }
        }
        $b = $this->randX($a, 1);
        for ($i = 0; $i < $a; $i++) {
            $b[$i][0] = $b[$i][0] * 2.0 - 1.0;
        }
        return array($w, $b);
    }
    
    protected function tansig($n, $b)
    {
        $n = Math::add($n, $b);
        $a = Math::scalarAdd(-1, Math::scalarDivide(2.0, Math::scalarAdd(1.0, Math::arrayExp(Math::scalarMultiply(-2.0, $n)))));
        list($rows, $columns) = Math::getShape($a);
        for ($x = 0; $x < $rows; $x++) {
            for ($y = 0; $y < $columns; $y++) {
                $v = $a[$x][$y];
                if (is_infinite(abs($v))) {
                    $a[$x][$y] = Math::sign($n[$x][$y]);
                }
            }
        }
        return $a;
    }

    protected function purelin($n, $b)
    {
        $a = Math::add($n, $b); //suma tablic 2d
        return $a;
    }

    protected function sumsqr($a)
    {
        list($rows, $columns) = Math::getShape($a);
        $sumSq = 0;
        for ($x = 0; $x < $rows; $x++) {
            for ($y = 0; $y < $columns; $y++) {
                $v = $a[$x][$y];
                $sumSq += pow($v, 2.0);
            }
        }
        return $sumSq;
    }

    protected function deltalin($a, $d)
    {
        return $d;
    }

    protected function deltatan($a, $d, ...$w) //funkcja zwiazana z propagacja wsteczna
    {
        //funkcja oblicza wartosci delta (bledy popelniane) dla sieci z tangensem hiperbolicznym jako funkcja aktywacji
        //Dla warstwy wyjsciowej uzywa sie deltain (roznica miedzy wyjsciem a wartosciami pozadanymi)
        //Dla warstw ukrytych uzywa sie reguly delta i wartosci delta z nastepnej warstwy
        $aModified = Math::scalarSubtract(Math::arrayMultiplication($a, $a), 1, true);
        if (empty($w)) {
            $d = Math::arrayMultiplication($aModified, $d); //(1 - (a^2)) * delta_next
        } else {
            //transpose dopasowuje macierz wag do wymiarow wartosci delta
            $d = Math::arrayMultiplication($aModified, Math::dot(Math::transpose($w[0]), $d)); //delta = (1 - (a^2)) * (transpose(w) * delta_next)
        }
        return $d;
    }
    //Oblicza zmiane wag i biasow
    protected function learnbp($p, $d, $lr) //p wejscie do obecnej warstwy, d-delta 
    {
        $x = Math::scalarMultiply($lr, $d); //wymnozenie lr przez delte
        $dw = Math::dot($x, Math::transpose($p)); //kalkuluje zmiane wagi ktora zmniejszy blad sieci
        $Q = Math::getShape($p)[1];
        $db = Math::dot($x, Math::staticNumberArray(1, $Q, 1)); //kalkuluje zmiane biasu ktora zmniejszy blad sieci
        return array($dw, $db);
    }

    //tworzy 2d array o ksztalcie takim jak n, kazdy wiersz ma wszystkie kolumny z ta sama wartoscia wektora b
    //Liczba wierszy zalezy od K1, liczba kolumn od liczby wierszy x
    protected function fillBiasArray($b, $n)
    {
        $result = [];
        for ($i = 0; $i < count($b); $i++) {
            $result[] = array_fill(0, count($n[0]), $b[$i][0]);
        }
        return $result;
    }

    protected function prepareInput(array $x): array
    {
        $x = Math::transpose($x);
        //Konwersja do float
        $x = array_map(function ($row) {
            return array_map('floatval', $row);
        }, $x);

        $x_norm = [];
        $x_norm_min = -1;
        $x_norm_max = 1;
        $epsilon = 0.00001;

        //Normalizacja wartości x
        for ($i = 0; $i < count($x); $i++) {
            $x_norm[$i] = [];
            $x_min = min($x[$i]);
            $x_max = max($x[$i]);

            for ($j = 0; $j < count($x[0]); $j++) {
                $x_norm[$i][] = ($x_norm_max - $x_norm_min) / ($x_max - $x_min + $epsilon) * ($x[$i][$j] - $x_min) + $x_norm_min;
            }
        }
        return $x_norm;
    }

    protected function calculatePK(array $result, array $y): float
    {
        $number_of_true = 0;
        foreach ($result[0] as $index => $value) {
            if (abs($value - $y[0][$index]) < 0.5)
                $number_of_true++;
        }

        return ($number_of_true / count($y[0])) * 100;
    }
}
