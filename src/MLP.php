<?php

namespace NeuralNetwork;

class MLP extends NNet
{
    private $lr;
    private $err_goal;
    private $ksi_inc;
    private $ksi_dec;
    private $er;
    private $max_epoch;
    private $w1;
    private $b1;
    private $w2;
    private $b2;
    private $w3;
    private $b3;
    private $SSE;

    private $y1;
    private $y2;
    private $y3;

    private $e;
    private $SSE_t_1;
    private $PK;
    private $d3;
    private $d2;
    private $d1;
    private $dw1;
    private $db1;
    private $dw2;
    private $db2;
    private $dw3;
    private $db3;
    private $mc;
    private $K1;
    private $K2;
    private $K3;
    private $L;
    private $x;

    private function initValues(): void
    {
        list($this->w1, $this->b1) = $this->nwtan($this->K1, $this->L);
        list($this->w2, $this->b2) = $this->nwtan($this->K2, $this->K1);
        list($this->w3, $this->b3) = $this->rands($this->K3, $this->K2);
        $this->SSE = 0;
    }

    public function __construct(int $l_input, int $max_epoch = 1000, int $K1 = 8, int $K2 = 6, int $K3_output = 1, float $lr = 0.00001, float $err_goal = 0.1, float $mc = 0.7, float $ksi_inc = 1.05, float $ksi_dec = 0.7, float $er = 1.04)
    {
        $this->lr = $lr;
        $this->err_goal = $err_goal;
        $this->ksi_inc = $ksi_inc;
        $this->ksi_dec = $ksi_dec;
        $this->er = $er;
        $this->max_epoch = $max_epoch;
        $this->mc = $mc;
        $this->L = $l_input;
        $this->K1 = $K1;
        $this->K2 = $K2;
        $this->K3 = $K3_output;
        //inicjalizacja wag i biasow
        $this->initValues();
    }

    private function calculatePrediction(array $x)
    {
        $n = Math::dot($this->w1, $x);
        $this->y1 = $this->tansig($n, $this->fillBiasArray($this->b1, $n));
        $n = Math::dot($this->w2, $this->y1);
        $this->y2 = $this->tansig($n, $this->fillBiasArray($this->b2, $n));
        $n = Math::dot($this->w3, $this->y2);
        $this->y3 = $this->purelin($n, $this->fillBiasArray($this->b3, $n));
        return $this->y3;
    }

    //Funkcja zwraca przewidywane wartosci na podstawie danych wejsciowych
    public function predict(array $x): array
    {
        if (is_null($x)) {
            throw new \Exception("Weights should be loaded or model should be trained before using predict");
        }
        if (count($x) == count($x, COUNT_RECURSIVE))
            $x = [$x];
        $this->normaliseWithTrainingData($x);
        return $this->calculatePrediction($x);
    }

    public function getModel(): array
    {
        return [
            'w1' => $this->w1,
            'b1' => $this->b1,
            'w2' => $this->w2,
            'b2' => $this->b2,
            'w3' => $this->w3,
            'b3' => $this->b3,
            'x' => $this->x
        ];
    }

    public function loadModel(array $data): void
    {
        $this->w1 = $data['w1'];
        $this->b1 = $data['b1'];
        $this->w2 = $data['w2'];
        $this->b2 = $data['b2'];
        $this->w3 = $data['w3'];
        $this->b3 = $data['b3'];
        $this->x = $data['x'];
    }

    public function loadSavedModel(string $path = 'example.json'): void
    {
        $data = json_decode(file_get_contents($path), true);
        $this->loadModel($data);
    }

    public function saveModel(string $path = 'example.json'): void
    {
        file_put_contents($path, json_encode($this->getModel(), JSON_PRETTY_PRINT));

    }

    public function roundOutput(array &$output): void
    {
        foreach ($output as &$row) {
            foreach ($row as &$value) {
                $value = round($value);
            }
        }
    }

    private function normaliseWithTrainingData(array &$x): void
    {
        $mergedData = array_merge($this->x, $x);
        $result = $this->prepareInput($mergedData);
        $mergedNormalised = Math::transpose($result);
        $normalisedInput = array_slice($mergedNormalised, -count($x));
        $x = Math::transpose($normalisedInput);
    }

    public function crossValidationTrain(array $x, array $y, int $CVN = 10, bool $print_results = true): float
    {
        $target = $y;
        $PK_vec = [];

        $indices = range(0, count($x) - 1); //tworzy tablice indexow o takim samym rozwmiarze jak ilosc wierszy danych
        shuffle($indices); //losowe wymieszanie kolejnosci elementow tablicy

        $fold_size = intval(count($x) / $CVN); //obliczenie rozmiaru podzialu
        for ($i = 0; $i < $CVN; $i++) {
            $start = $i * $fold_size;
            $end = ($i + 1) * $fold_size;

            $test_indices = array_slice($indices, $start, $fold_size); //wybranie indexow do danych testowych
            $train_indices = array_merge(array_slice($indices, 0, $start), array_slice($indices, $end)); //wybranie indexow do danych treningowych (wszystkie z wyjatkiem danych testowych)
            $x_train = [];
            $x_test = [];
            $y_train = [];
            $y_test = [];

            foreach ($train_indices as $train_indice) {
                $x_train[] = $x[$train_indice];
                $y_train[] = [$target[0][$train_indice]];
            }

            foreach ($test_indices as $test_indice) {
                $x_test[] = $x[$test_indice];
                $y_test[] = [$target[0][$test_indice]];
            }
            $this->initValues();
            $this->train($x_train, Math::transpose($y_train));
            $this->normaliseWithTrainingData($x_test);
            $result = $this->calculatePrediction($x_test);
            $number_of_samples = count($y_test); //Obliczenie liczby probek danych testowych
            $PK_vec[] = $this->calculatePK($result, Math::transpose($y_test));
            if ($print_results)
                echo "Test #" . ($i + 1) . ": PK_vec " . $PK_vec[$i] . " test_size " . $number_of_samples . "\n";
        }

        $pkAverage = array_sum($PK_vec) / count($PK_vec);
        if ($print_results)
            echo "Średnie PK: $pkAverage\n";
        return $pkAverage;
    }

    public function train(array $x_train, array $y_train): void
    {
        $this->x = $x_train;
        $x_train = $this->prepareInput($x_train);
        $w1_t_1 = $this->w1;
        $b1_t_1 = $this->b1;
        $w2_t_1 = $this->w2;
        $b2_t_1 = $this->b2;
        $w3_t_1 = $this->w3;
        $b3_t_1 = $this->b3;
        for ($epoch = 1; $epoch <= $this->max_epoch; $epoch++) {
            $this->y3 = $this->calculatePrediction($x_train);
            $this->e = Math::subtract($y_train, $this->y3);
            $this->SSE_t_1 = $this->SSE;
            $this->SSE = $this->sumsqr($this->e);
            $this->PK = $this->calculatePK($this->y3, $y_train);
            if ($this->SSE < $this->err_goal || $this->PK == 100)
                break; //szkolenie jest przerywane gdy blad bedzie wystarczajaco maly lub PK = 100
            if (is_nan($this->SSE))
                break; //Jeśli SSE nie jest liczba trenowanie jest przerywane

            if ($this->SSE > $this->er * $this->SSE_t_1)
                $this->lr *= $this->ksi_dec; //Jezeli blad sie zwiekszyl zmniejsz krok lr
            else if ($this->SSE < $this->SSE_t_1)
                $this->lr *= $this->ksi_inc; //Jezeli blad sie zmniejszyl zwieksz krok lr

            //Propagacja wsteczna (obliczanie wartosci gradientu dla danej warstwy)
            $this->d3 = $this->deltalin($this->y3, $this->e);
            $this->d2 = $this->deltatan($this->y2, $this->d3, $this->w3);
            $this->d1 = $this->deltatan($this->y1, $this->d2, $this->w2);
            //learnbp oblicza zmianę wag i biasow poszczegolnych warstw
            list($this->dw1, $this->db1) = $this->learnbp($x_train, $this->d1, $this->lr);
            list($this->dw2, $this->db2) = $this->learnbp($this->y1, $this->d2, $this->lr);
            list($this->dw3, $this->db3) = $this->learnbp($this->y2, $this->d3, $this->lr);

            $w1_temp = $this->w1;
            $b1_temp = $this->b1;
            $w2_temp = $this->w2;
            $b2_temp = $this->b2;
            $w3_temp = $this->w3;
            $b3_temp = $this->b3;
            //Aktualizacja wag i biasow w poszczegolnych warstwach sieci
            $this->w1 = Math::add($this->w1, Math::add($this->dw1, Math::scalarMultiply($this->mc, Math::subtract($this->w1, $w1_t_1))));
            $this->b1 = Math::add($this->b1, Math::add($this->db1, Math::scalarMultiply($this->mc, Math::subtract($this->b1, $b1_t_1))));
            $this->w2 = Math::add($this->w2, Math::add($this->dw2, Math::scalarMultiply($this->mc, Math::subtract($this->w2, $w2_t_1))));
            $this->b2 = Math::add($this->b2, Math::add($this->db2, Math::scalarMultiply($this->mc, Math::subtract($this->b2, $b2_t_1))));
            $this->w3 = Math::add($this->w3, Math::add($this->dw3, Math::scalarMultiply($this->mc, Math::subtract($this->w3, $w3_t_1))));
            $this->b3 = Math::add($this->b3, Math::add($this->db3, Math::scalarMultiply($this->mc, Math::subtract($this->b3, $b3_t_1))));

            $w1_t_1 = $w1_temp;
            $b1_t_1 = $b1_temp;
            $w2_t_1 = $w2_temp;
            $b2_t_1 = $b2_temp;
            $w3_t_1 = $w3_temp;
            $b3_t_1 = $b3_temp;
        }
    }
}
