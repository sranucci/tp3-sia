# Trabajo Práctico 3 - SIA

## Requisitos:
1) Python 3
2) Pip 3


## Ejecución:
1) Ejecutar en una linea de comandos (con pip instalado):
```
pip install -r requirements.txt
```
2) Dependiendo de que ejercicio se quiere ejecutar, debe entrar en el directorio correspondiente y alterar el config.json correspondiente (se describe la configuración abajo).
3) Para cada ejercicio, ejecutar el main.py correspondiente. Ejemplo:
```
python ./ex2/main.py
```

## Configuración Ejercicio 2
| Campo             | Descripción                                                                    | Valores aceptados                                                            |  
|-------------------|--------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| seed              | Permite especificar el numero inicial usado al generar los valores aleatorios. | Debe entero positivo. Tambien se puede usar -1 si no se quiere usar un seed. |
| training_data     | Path relativo de los datos usados en el entrenamiento.                         | Cadena de caracteres.                                                        |
| learning_constant | Valor multiplicador usado en la actualización de los pesos.                    | Numero de punto flotante entre 0 y 1.                                        |
| limit             | Maxima cantidad de iteraciones al entrenar.                                    | Numero entero mayor a 0.                                                     | 
| epsilon           | Valor de error minimo acceptable.                                              | Numero de punto flotante o entero mayor o igual a 0.                         | 
| method            | Metodo de aprendizaje.                                                         | Objeto json con configuración. [ Ver tabla 1 ]                               | 
| selection_method  | Metodo de cross-validation                                                     | Objeto json con configuración. [ Ver tabla 2 ]                               | 

## Tabla 1: Metodo de Aprendizaje
| Campo | Descripción                                                                     | Valores aceptados                 |  
|-------|---------------------------------------------------------------------------------|-----------------------------------|
| type  | Especifica si se quiere usar un metodo lineal o no lineal.                      | "linear", "non_linear"            |
| theta | Solo para "non_linear". Especifica función de activación.                       | "logistic", "tanh"                | 
| beta  | Solo para "non_linear". Especifica el valor de beta usando en la funcion theta. | Numero de punto flotante mayor 0. | 


## Tabla 2: Metodo de Selección de Cross-Validation
| Campo      | Descripción                                                                        | Valores aceptados                                      |  
|------------|------------------------------------------------------------------------------------|--------------------------------------------------------|
| type       | Especifica que metodo de selección de cross-validation se quiere usar.             | "simple", "k-fold"                                     |
| proportion | Solo para "simple". Especifica que proporción de los datos se elige para entrenar. | Numero de punto flotante mayor a 0, menor a 1          | 
| folds      | Solo para "k-fold". Especifica la cantidad de folds que se generan.                | Numero entero mayor a 0, menor a la cantidad de datos. | 


## Configuración Ejercicio 3 C
| Campo                    | Descripción                                                                    | Valores aceptados                                                            |  
|--------------------------|--------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| seed                     | Permite especificar el numero inicial usado al generar los valores aleatorios. | Debe entero positivo. Tambien se puede usar -1 si no se quiere usar un seed. |
| training_data_input      | Path relativo de los datos de entrada de entrenamiento.                        | Cadena de caracteres.                                                        |
| training_data_output     | Path relativo de los datos de salida de entrenamiento.                         | Cadena de caracteres.                                                        |
| selection_method         | Metodo de cross-validation                                                     | Objeto json con configuración. [ Ver tabla 3 ]                               | 
| max_noise                | Ruido maximo para aplicarse sobre los datos de testeo.                         | Numero de punto flotante entre 0 y 1.                                        | 
| activation_function_beta | Valor de beta para la funcion logistica.                                       | Numero de punto flotante o entero mayor o igual a 0.                         | 
| learning_constant        | Valor multiplicador usado en la actualización de los pesos.                    | Numero de punto flotante entre 0 y 1.                                        |
| limit                    | Maxima cantidad de iteraciones al entrenar.                                    | Numero entero mayor a 0.                                                     | 
| epsilon                  | Valor de error minimo acceptable.                                              | Numero de punto flotante o entero mayor o igual a 0.                         |
| batch_size               | Tamano de datos para usar al entrenar en cada epoca.                           | Numero entero entre 0 y la cantidad de datos que se usan para entrenar       |
| hidden_layer_amount      | Cantidad de capas intermedias                                                  | Numero entero mayor a 0.                                                     | 
| neurons_per_layer        | Cantidad de nueronas por capa intermedia                                       | Numero entero mayor a 0.                                                     | 
| optimization_method      | Metodo de optimizacion                                                         | Objeto json con configuración. [ Ver tabla 4]                                |

## Tabla 3: Metodo de Selección de Cross-Validation
| Campo      | Descripción                                                                        | Valores aceptados                                      |  
|------------|------------------------------------------------------------------------------------|--------------------------------------------------------|
| type       | Especifica que metodo de selección de cross-validation se quiere usar.             | "simple", "none"                                       |
| proportion | Solo para "simple". Especifica que proporción de los datos se elige para entrenar. | Numero de punto flotante mayor a 0, menor a 1          | 
| folds      | Solo para "k-fold". Especifica la cantidad de folds que se generan.                | Numero entero mayor a 0, menor a la cantidad de datos. | 

## Tabla 4: Metodo de Optimizacion
| Campo | Descripción                                                                       | Valores aceptados                           |  
|-------|-----------------------------------------------------------------------------------|---------------------------------------------|
| type  | Especifica que metodo de optimizacion se quiere usar.                             | "momentum"                                  |
| alpha | Solo para "momentum". Especifica la ponderacion que se le da al delta W anterior. | Numero de punto flotante mayor o igual a 0. |
