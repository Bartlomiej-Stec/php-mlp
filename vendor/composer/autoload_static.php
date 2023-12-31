<?php

// autoload_static.php @generated by Composer

namespace Composer\Autoload;

class ComposerStaticInitf5ab4e1a133afd6ca789218df78a66c1
{
    public static $prefixLengthsPsr4 = array (
        'N' => 
        array (
            'NeuralNetwork\\' => 14,
        ),
    );

    public static $prefixDirsPsr4 = array (
        'NeuralNetwork\\' => 
        array (
            0 => __DIR__ . '/../..' . '/src',
        ),
    );

    public static $classMap = array (
        'Composer\\InstalledVersions' => __DIR__ . '/..' . '/composer/InstalledVersions.php',
    );

    public static function getInitializer(ClassLoader $loader)
    {
        return \Closure::bind(function () use ($loader) {
            $loader->prefixLengthsPsr4 = ComposerStaticInitf5ab4e1a133afd6ca789218df78a66c1::$prefixLengthsPsr4;
            $loader->prefixDirsPsr4 = ComposerStaticInitf5ab4e1a133afd6ca789218df78a66c1::$prefixDirsPsr4;
            $loader->classMap = ComposerStaticInitf5ab4e1a133afd6ca789218df78a66c1::$classMap;

        }, null, ClassLoader::class);
    }
}
