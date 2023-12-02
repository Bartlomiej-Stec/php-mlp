<?php

namespace NeuralNetwork;

class Optimizer
{
    private $csv_repeats;
    private $csv_folds;

    public function __construct(int $csv_repeats = 3, int $csv_folds = 10)
    {
        $this->csv_repeats = $csv_repeats;
        $this->csv_folds = $csv_folds;
    }

    public function findOptimalK(array $x, array $y, int $maxK1, int $maxK2, int $stepK1, int $stepK2, int $max_epoch = 1000, int $K3_output = 1, float $lr = 0.00001, float $err_goal = 0.1, float $mc = 0.7, float $ksi_inc = 1.05, float $ksi_dec = 0.7, float $er = 1.04)
    {
        $K1_list = range(2, $maxK1, $stepK1);
        $K2_list = range(2, $maxK2, $stepK2);
        $sumTests = count($K1_list) * count($K2_list);
        $bestParams = [];
        $bestScorage = 0;
        $index = 1;
        foreach ($K1_list as $K1) {
            foreach ($K2_list as $K2) {
                $partialResult = [];
                for ($i = 0; $i < $this->csv_repeats; $i++) {
                    $mlp = new MLP(count($x[0]), $max_epoch, $K1, $K2, $K3_output, $lr, $err_goal, $mc, $ksi_inc, $ksi_dec, $er);
                    $partialResult[] = $mlp->crossValidationTrain($x, $y, $this->csv_folds, false);
                }
                $medianScore = Math::calculateMedian($partialResult);
                if ($medianScore > $bestScorage) {
                    $bestScorage = $medianScore;
                    $bestParams['K1'] = $K1;
                    $bestParams['K2'] = $K2;
                }
                echo "TEST $index/$sumTests. Best params: K1: {$bestParams['K1']}, K2: {$bestParams['K2']}, Accuracy: $bestScorage\n";
                $index++;
            }
        }
    }

    public function findOptimalLrEpochErr(array $x, array $y, array $max_epoch = [], array $lr = [], array $err_goal = [], int $K1 = 8, int $K2 = 6, int $K3_output = 1, float $mc = 0.7, float $ksi_inc = 1.05, float $ksi_dec = 0.7, float $er = 1.04)
    {
        if ($max_epoch == [])
            $max_epoch = [1000, 500, 2000, 5000, 10000];
        if ($lr == [])
            $lr = [0.0001, 0.001, 0.000001];
        if ($err_goal == [])
            $err_goal = [0.1, 0.25, 0.001];
        $sumTests = count($err_goal) * count($lr) * count($max_epoch);
        $bestParams = [];
        $bestScorage = 0;
        $index = 1;
        foreach ($max_epoch as $currentEpoch) {
            foreach ($err_goal as $currentErr) {
                foreach ($lr as $currentLr) {
                    $partialResult = [];
                    for ($i = 0; $i < $this->csv_repeats; $i++) {
                        $mlp = new MLP(count($x[0]), $currentEpoch, $K1, $K2, $K3_output, $currentLr, $currentErr, $mc, $ksi_inc, $ksi_dec, $er);
                        $partialResult[] = $mlp->crossValidationTrain($x, $y, $this->csv_folds, false);
                    }
                    $medianScore = Math::calculateMedian($partialResult);
                    if ($medianScore > $bestScorage) {
                        $bestScorage = $medianScore;
                        $bestParams['max_epoch'] = $currentEpoch;
                        $bestParams['lr'] = $currentLr;
                        $bestParams['err_goal'] = $currentErr;
                    }
                    echo "TEST $index/$sumTests. Best params: max_epoch: {$bestParams['max_epoch']}, lr: {$bestParams['lr']}, err_goal: {$bestParams['err_goal']}, Accuracy: $bestScorage\n";
                    $index++;
                }
            }
        }
    }
}