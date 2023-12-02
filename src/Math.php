<?php

namespace NeuralNetwork;

use Exception;

class Math
{
    public static function getShape($n)
    {
        return array(count($n), count($n[0]));
    }

    public static function scalarMultiply($a, $b)
    {
        $c = array();
        for ($i = 0; $i < count($b); $i++) {
            $row = array();
            for ($j = 0; $j < count($b[$i]); $j++) {
                $row[] = $a * $b[$i][$j];
            }
            $c[] = $row;
        }
        return $c;
    }

    public static function staticNumberArray($number, $rows, $cols)
    {
        $result = array();
        for ($i = 0; $i < $rows; $i++) {
            $result[$i] = array();
            for ($j = 0; $j < $cols; $j++) {
                $result[$i][$j] = $number;
            }
        }
        return $result;
    }

    public static function transpose($a)
    {
        $rows = count($a);
        $cols = count($a[0]);

        $result = array();
        for ($i = 0; $i < $cols; $i++) {
            $result[$i] = array();
            for ($j = 0; $j < $rows; $j++) {
                $result[$i][$j] = $a[$j][$i];
            }
        }

        return $result;
    }

    public static function subtract($a, $b)
    {
        $c = array();
        for ($i = 0; $i < count($a); $i++) {
            $row = array();
            for ($j = 0; $j < count($a[$i]); $j++) {
                $row[] = $a[$i][$j] - $b[$i][$j];
            }
            $c[] = $row;
        }
        return $c;
    }

    public static function arrayDivide($a, $b)
    {
        $rows = count($a);
        $cols = count($a[0]);

        for ($i = 0; $i < $rows; $i++) {
            for ($j = 0; $j < $cols; $j++) {
                $a[$i][$j] /= $b[$i][$j];
            }
        }
        return $a;
    }

    public static function sign($a)
    {
        if (is_array($a)) {
            $c = array();
            for ($i = 0; $i < count($a); $i++) {
                $row = array();
                for ($j = 0; $j < count($a[$i]); $j++) {
                    $row[] = self::sign($a[$i][$j]);
                }
                $c[] = $row;
            }
            return $c;
        } else {
            return ($a > 0 ? 1 : ($a < 0 ? -1 : 0));
        }
    }

    public static function add($a, $b)
    {
        $c = array();
        for ($i = 0; $i < count($a); $i++) {
            $row = array();
            for ($j = 0; $j < count($a[$i]); $j++) {
                $row[] = $a[$i][$j] + $b[$i][$j];
            }
            $c[] = $row;
        }
        return $c;
    }

    public static function scalarAdd($a, $b)
    {
        $c = array();
        for ($i = 0; $i < count($b); $i++) {
            $row = array();
            for ($j = 0; $j < count($b[$i]); $j++) {
                $row[] = $a + $b[$i][$j];
            }
            $c[] = $row;
        }
        return $c;
    }

    public static function scalarDivide($a, $b)
    {
        $c = array();
        for ($i = 0; $i < count($b); $i++) {
            $row = array();
            for ($j = 0; $j < count($b[$i]); $j++) {
                $row[] = $a / $b[$i][$j];
            }
            $c[] = $row;
        }
        return $c;
    }

    public static function arrayExp($a)
    {
        $c = array();
        for ($i = 0; $i < count($a); $i++) {
            $row = array();
            for ($j = 0; $j < count($a[$i]); $j++) {
                $row[] = exp($a[$i][$j]);
            }
            $c[] = $row;
        }
        return $c;
    }

    public static function arrayMultiplication($a, $b)
    {
        $rows = count($a);
        $cols = count($a[0]);

        for ($i = 0; $i < $rows; $i++) {
            for ($j = 0; $j < $cols; $j++) {
                $a[$i][$j] *= $b[$i][$j];
            }
        }
        return $a;
    }

    public static function scalarSubtract($a, $minus, $reverse = false)
    {
        $rows = count($a);
        $cols = count($a[0]);

        for ($i = 0; $i < $rows; $i++) {
            for ($j = 0; $j < $cols; $j++) {
                if (!$reverse)
                    $a[$i][$j] = $a[$i][$j] - $minus;
                else
                    $a[$i][$j] = $minus - $a[$i][$j];
            }
        }
        return $a;
    }

    public static function dot($a, $b)
    {
        $m = count($a);
        $n = count($b);
        $p = count($b[0]);

        if (count($a[0]) != $n) {
            throw new Exception('Invalid matrix dimensions: A.columns != B.rows');
        }

        $c = array();
        for ($i = 0; $i < $m; $i++) {
            $c[$i] = array();
            for ($j = 0; $j < $p; $j++) {
                $sum = 0;
                for ($k = 0; $k < $n; $k++) {
                    $sum += $a[$i][$k] * $b[$k][$j];
                }
                $c[$i][$j] = $sum;
            }
        }

        return $c;
    }

    public static function calculateMedian($array)
    {
        sort($array);
        $count = count($array);
        $middleval = floor(($count - 1) / 2);
        if ($count % 2) {
            $median = $array[$middleval];
        } else {
            $low = $array[$middleval];
            $high = $array[$middleval + 1];
            $median = (($low + $high) / 2);
        }
        return $median;
    }
}


