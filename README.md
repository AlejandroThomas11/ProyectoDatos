# Análisis temático de proyectos de investigación

**Asignatura:** Procesamiento de Datos  
**Titulación:** Máster en Ingeniería de Telecomunicación

## 1. Introducción

Este proyecto tiene como objetivo estudiar diferentes técnicas de Machine Learning para clasificar automáticamente proyectos de investigación según su temática científica.

Para ello, se utiliza un conjunto de datos de proyectos financiados por la Comisión Europea dentro del programa Horizonte 2020 (H2020). La clasificación se realiza a partir de los resúmenes de los proyectos, utilizando las categorías temáticas definidas en la taxonomía EuroSciVoc.

A lo largo del trabajo se comparan distintas formas de representar el texto, como TF-IDF, Word2Vec y embeddings obtenidos mediante Transformers, junto con diferentes modelos de clasificación, incluyendo SVM y redes neuronales.

La evaluación se realiza mediante las métricas accuracy y F1-macro. Además, se analizan los errores de clasificación, la incertidumbre de los resultados y la influencia de los metadatos administrativos mediante una extensión adicional.

El objetivo final es comparar el rendimiento de las diferentes alternativas y estudiar qué técnicas resultan más adecuadas para identificar la temática de los proyectos de investigación.

## 2. Conjunto de datos y preparación

### 2.1. Descripción del conjunto de datos

El conjunto de datos utilizado corresponde a proyectos de investigación financiados por la Comisión Europea mediante el programa Horizonte 2020 (H2020). La información original se encuentra en el archivo `projects.xlsx`, que contiene 35.378 proyectos.

Para cada proyecto se dispone de su identificador, título, resumen, fechas de ejecución, información económica, países participantes y códigos de clasificación científica. El archivo `SciVocCodes.xlsx` permite relacionar estos códigos con las categorías de la taxonomía EuroSciVoc.

El objetivo del trabajo es predecir la categoría científica de segundo nivel a partir del resumen de cada proyecto.

### 2.2. Selección y clasificación de los proyectos

Como algunos proyectos tienen asociados varios códigos EuroSciVoc, se decidió utilizar el primer código válido de cada proyecto que permitiera identificar una categoría de segundo nivel.

Se descartaron los registros para los que no fue posible obtener una categoría válida. De los 35.378 proyectos originales, se conservaron 31.982, distribuidos en 39 categorías de segundo nivel, pertenecientes a 6 áreas generales de primer nivel.

### 2.3. División de los datos

Los proyectos se dividieron en tres conjuntos independientes:

| Conjunto | Número de proyectos | Porcentaje aproximado |
|---|---:|---:|
| Entrenamiento | 19.188 | 60 % |
| Validación | 6.397 | 20 % |
| Prueba | 6.397 | 20 % |
| **Total** | **31.982** | **100 %** |

Se utilizó `StratifiedGroupKFold` de Scikit-learn, agrupando los proyectos que compartían resúmenes idénticos para evitar que aparecieran en diferentes conjuntos. También se buscó mantener una distribución adecuada de las categorías.

El conjunto de entrenamiento se utilizó para ajustar los modelos, el de validación para comparar configuraciones y el de prueba para evaluar los resultados finales.

### 2.4. Preprocesamiento del texto

Para las representaciones basadas en TF-IDF y Word2Vec se realizó un procesamiento de los resúmenes mediante spaCy y su modelo de inglés `en_core_web_sm`.

Se eliminaron las palabras vacías y los elementos que no estaban formados únicamente por letras. Además, se aplicó lematización, se convirtieron los términos a minúsculas y se conservaron únicamente las palabras con al menos tres caracteres.

Este procesamiento permitió reducir las variaciones de las palabras y preparar los documentos para su representación numérica.

Para los modelos basados en Transformers se conservaron los resúmenes originales, utilizando la tokenización correspondiente al modelo preentrenado.

## 3. Metodología y modelos de clasificación

Para estudiar la influencia de las distintas representaciones del texto, se han entrenado varios modelos utilizando las mismas particiones de entrenamiento, validación y prueba.

El rendimiento se ha evaluado mediante accuracy y F1-macro. Esta segunda métrica resulta especialmente importante porque las 39 categorías no tienen el mismo número de proyectos y permite valorar el comportamiento del clasificador en las clases minoritarias.

### 3.1. Modelo de referencia

En primer lugar, se estableció un modelo de referencia que asigna siempre la categoría más frecuente del conjunto de entrenamiento.

Este modelo permite comprobar si las técnicas utilizadas ofrecen una mejora respecto a una estrategia sencilla que no tiene en cuenta el contenido de los resúmenes.

### 3.2. TF-IDF y SVM lineal

La primera representación utilizada fue TF-IDF, que asigna pesos a los términos en función de su frecuencia dentro de cada documento y de su presencia en el conjunto de entrenamiento.

Se utilizó `TfidfVectorizer` de Scikit-learn con las siguientes características:

- Palabras individuales y pares de palabras (*unigramas y bigramas*).
- Frecuencia mínima de aparición en 3 documentos.
- Frecuencia máxima del 95 % de los documentos.
- Un máximo de 50.000 características.
- Frecuencia de términos sublineal y normalización L2.

El vocabulario y los pesos TF-IDF se ajustaron únicamente con los datos de entrenamiento.

Posteriormente, se entrenó un clasificador SVM lineal mediante `LinearSVC`. Para seleccionar su configuración, se compararon valores del parámetro de regularización `C` de 0,1, 1 y 10, tanto con pesos balanceados como sin ellos.

La selección se realizó utilizando el F1-macro del conjunto de validación.

### 3.3. Word2Vec y SVM lineal

Como segunda representación, se utilizó Word2Vec para obtener vectores que representaran las relaciones entre las palabras.

El modelo se entrenó con los textos procesados del conjunto de entrenamiento, utilizando vectores de 200 dimensiones, una ventana de contexto de 5 palabras, una frecuencia mínima de 5 apariciones y el algoritmo Skip-gram durante 5 épocas.

Para representar cada proyecto, se calculó la media de los vectores de las palabras conocidas presentes en su resumen.

A continuación, los vectores obtenidos se estandarizaron mediante `StandardScaler`, ajustado únicamente sobre el conjunto de entrenamiento.

Finalmente, se entrenó un SVM lineal y se compararon diferentes configuraciones de regularización y ponderación de clases, seleccionando la mejor mediante el F1-macro de validación.

### 3.4. Embeddings de DistilBERT y SVM

Como tercera representación del texto, se utilizó el modelo preentrenado `distilbert-base-uncased` de Hugging Face.

En este primer experimento, DistilBERT se empleó como extractor de características, manteniendo congelados sus parámetros. Por tanto, no se modificaron sus pesos durante la obtención de los embeddings.

Los resúmenes originales se procesaron mediante el tokenizador de DistilBERT, con una longitud máxima de 512 tokens. Para obtener una representación de cada proyecto, se calculó la media de los vectores de los tokens válidos, teniendo en cuenta la máscara de atención.

De este modo, cada proyecto quedó representado mediante un vector de 768 dimensiones.

Los embeddings se estandarizaron utilizando `StandardScaler`, ajustado exclusivamente sobre los datos de entrenamiento. Posteriormente, se entrenó un SVM lineal y se compararon configuraciones con y sin ponderación balanceada de clases, utilizando el F1-macro de validación para seleccionar el modelo.

### 3.5. Red neuronal con PyTorch

A partir de los embeddings obtenidos con DistilBERT, se desarrolló una red neuronal mediante PyTorch para comprobar si un clasificador neuronal podía mejorar los resultados del SVM.

La arquitectura utilizada fue una red neuronal completamente conectada, con las siguientes capas:

- Capa de entrada de 768 características.
- Capa oculta de 256 neuronas.
- Función de activación ReLU.
- Capa Dropout con probabilidad de 0,3.
- Capa de salida de 39 neuronas, una por categoría científica.

Para el entrenamiento se utilizó la función de pérdida `CrossEntropyLoss` y el optimizador Adam, con una tasa de aprendizaje de 0,001 y lotes de 64 proyectos.

Inicialmente se entrenó durante 12 épocas y posteriormente se amplió el entrenamiento hasta completar 20. Después de cada época se evaluó el modelo utilizando el conjunto de validación y se guardaron los pesos cuando mejoraba el F1-macro.

La mejor configuración se obtuvo en la época 12. Se conservaron sus pesos para realizar posteriormente la evaluación sobre el conjunto de prueba.

### 3.6. Fine-tuning de DistilBERT

Finalmente, se realizó un ajuste fino (*fine-tuning*) de DistilBERT para adaptar directamente el modelo preentrenado a la clasificación de las 39 categorías científicas.

A diferencia del experimento anterior con embeddings congelados, en este caso se actualizaron los parámetros de DistilBERT durante el entrenamiento, junto con la capa de clasificación.

Se utilizó `AutoModelForSequenceClassification` y la herramienta `Trainer` de Hugging Face.

La configuración principal fue la siguiente:

| Parámetro | Valor |
|---|---|
| Modelo preentrenado | `distilbert-base-uncased` |
| Número de categorías | 39 |
| Épocas | 3 |
| Tamaño de lote de entrenamiento | 8 |
| Tamaño de lote de validación | 16 |
| Acumulación de gradientes | 2 |
| Tasa de aprendizaje | 0,00002 |
| Weight decay | 0,01 |
| Longitud máxima | 512 tokens |
| Precisión mixta | FP16 |
| Semilla aleatoria | 42 |

El modelo se evaluó al finalizar cada época, utilizando el F1-macro de validación como criterio de selección. Al terminar el entrenamiento, se recuperó y guardó la versión que había obtenido el mejor resultado.

Este experimento permitió comparar el uso de un Transformer como extractor fijo de características frente a su adaptación mediante fine-tuning.

## 4. Resultados experimentales

### 4.1. Comparación de los modelos

Una vez entrenados los distintos clasificadores, se evaluó su rendimiento sobre los conjuntos de validación y prueba. Para comparar los resultados se utilizaron las métricas accuracy y F1-macro.

Los resultados obtenidos fueron los siguientes:

| Modelo | Accuracy validación | F1-macro validación | Accuracy prueba | F1-macro prueba |
|---|---:|---:|---:|---:|
| TF-IDF + SVM | 0,5371 | 0,4127 | 0,5281 | 0,4072 |
| Word2Vec + SVM | 0,4627 | 0,2922 | 0,4407 | 0,2866 |
| DistilBERT congelado + SVM | 0,4116 | 0,2601 | 0,3958 | 0,2453 |
| DistilBERT + red PyTorch | 0,4557 | 0,3002 | 0,4432 | 0,2948 |
| DistilBERT con fine-tuning | 0,5656 | 0,3256 | 0,5556 | 0,3149 |

### 4.2. Análisis de los resultados

El modelo basado en TF-IDF y SVM lineal consiguió el mejor F1-macro, con un valor de 0,4072 en el conjunto de prueba. Esto indica que fue el modelo que ofreció un mejor equilibrio en la clasificación de las 39 categorías científicas.

Por otro lado, DistilBERT con fine-tuning obtuvo el mayor accuracy, alcanzando un 55,56 % en prueba. Sin embargo, su F1-macro fue inferior al de TF-IDF, lo que indica que su mayor proporción global de aciertos no se tradujo en un mejor comportamiento equilibrado entre todas las categorías.

El modelo Word2Vec + SVM obtuvo resultados intermedios, con un F1-macro de 0,2866 en prueba. Por su parte, la representación mediante embeddings de DistilBERT congelado y SVM alcanzó un F1-macro de 0,2453.

Al utilizar estos mismos embeddings como entrada de una red neuronal desarrollada con PyTorch, el F1-macro aumentó hasta 0,2948, superando al clasificador SVM basado en embeddings congelados.

En conjunto, los resultados muestran que una representación tradicional como TF-IDF, combinada con un SVM lineal, puede resultar muy competitiva para esta tarea. También se observa que el fine-tuning de DistilBERT mejora el accuracy, aunque no consigue superar a TF-IDF en F1-macro.

### 4.3. Selección del modelo

Teniendo en cuenta que las categorías del conjunto de datos presentan diferentes frecuencias, se considera el F1-macro como la métrica principal para comparar los clasificadores.

De acuerdo con los resultados de validación, se seleccionó TF-IDF + SVM como el modelo principal del proyecto. Su evaluación final sobre los 6.397 proyectos de prueba obtuvo un accuracy del 52,81 % y un F1-macro de 0,4072.

No obstante, el fine-tuning de DistilBERT se mantiene como una alternativa destacable por conseguir el mayor porcentaje global de clasificaciones correctas.

### 4.4. Análisis de errores según EuroSciVoc

Para estudiar los errores de clasificación, se calculó la matriz de confusión del modelo TF-IDF + SVM sobre el conjunto de prueba.

De los 6.397 proyectos evaluados, el modelo clasificó correctamente 3.378 y cometió 3.019 errores.

Aprovechando la estructura jerárquica de EuroSciVoc, se analizaron estos errores para comprobar si las categorías confundidas pertenecían a la misma rama científica de primer nivel.

Los resultados fueron los siguientes:

| Tipo de error | Proyectos | Porcentaje |
|---|---:|---:|
| Categorías de la misma rama | 998 | 33,06 % |
| Categorías de ramas diferentes | 2.021 | 66,94 % |
| **Total** | **3.019** | **100 %** |

Estos resultados indican que aproximadamente un tercio de los errores se produce entre categorías que comparten una misma área científica general, mientras que los restantes corresponden a confusiones entre áreas diferentes.

Entre las confusiones más frecuentes se encuentran las relacionadas con ciencias biológicas, medicina e ingeniería, que pueden compartir terminología en sus resúmenes.

### 4.5. Análisis de incertidumbre

Para estimar la incertidumbre de los resultados se utilizó el método bootstrap, realizando 1.000 repeticiones con muestreo aleatorio con reemplazo sobre el conjunto de prueba.

Para el modelo TF-IDF + SVM se obtuvieron los siguientes resultados:

| Métrica | Valor | Intervalo bootstrap del 95 % |
|---|---:|---|
| Accuracy | 0,5281 | [0,5152; 0,5399] |
| F1-macro | 0,4072 | [0,3857; 0,4255] |

Además, se realizó una comparación bootstrap pareada entre TF-IDF + SVM y DistilBERT con fine-tuning.

La diferencia de F1-macro fue de 0,0923 a favor de TF-IDF, con un intervalo del 95 % de [0,0707; 0,1113]. Como el intervalo no incluye el cero, los resultados proporcionan evidencia de una diferencia favorable a TF-IDF.

También se compararon Word2Vec + SVM y la red neuronal PyTorch. En este caso, la diferencia de F1-macro fue de 0,0082 a favor de PyTorch, con un intervalo del 95 % de [-0,0105; 0,0274].

Como este último intervalo incluye el cero, no se dispone de evidencia suficiente para afirmar que la diferencia observada entre ambos modelos sea consistente.

### 4.6. Comparación entre los niveles 1 y 2 de EuroSciVoc

Además de la clasificación principal en 39 categorías de segundo nivel, se estudió el rendimiento de diferentes representaciones al utilizar las 6 categorías generales de primer nivel de EuroSciVoc.

El objetivo de este experimento fue comprobar cómo influye el nivel de detalle de las categorías en la dificultad del problema de clasificación.

Los resultados obtenidos mediante F1-macro fueron los siguientes:

| Representación | F1-macro nivel 1 | F1-macro nivel 2 |
|---|---:|---:|
| TF-IDF + SVM | 0,6286 | 0,4072 |
| Word2Vec + SVM | 0,5491 | 0,2866 |
| DistilBERT congelado + SVM | 0,5874 | 0,2453 |

Los tres modelos obtuvieron un F1-macro superior al clasificar los proyectos en categorías de primer nivel.

Este comportamiento es razonable, ya que distinguir entre 6 áreas científicas generales resulta menos complejo que diferenciar entre 39 categorías específicas.

TF-IDF + SVM obtuvo el mejor F1-macro en ambos niveles. Sin embargo, el orden de las otras representaciones cambió: mientras que Word2Vec superó a DistilBERT congelado en el segundo nivel, DistilBERT obtuvo mejores resultados al utilizar las categorías generales del primer nivel.

Estos resultados muestran que el rendimiento relativo de las representaciones puede variar dependiendo del nivel de detalle de la clasificación.

## 5. Extensión: ¿Qué aporta el texto?

### 5.1. Objetivo

En esta extensión se estudió la importancia del contenido textual frente a los metadatos administrativos para clasificar la temática científica de los proyectos.

Para ello, se compararon tres enfoques: un clasificador que utiliza únicamente metadatos, otro basado exclusivamente en los resúmenes mediante TF-IDF y SVM, y un tercero que combina ambas fuentes de información.

Se mantuvieron las mismas particiones de entrenamiento, validación y prueba utilizadas en el resto del proyecto para poder comparar los resultados.

### 5.2. Preparación de los metadatos

Se utilizaron cinco características administrativas de cada proyecto:

- Coste total (`totalCost`).
- Financiación máxima de la Unión Europea (`ecMaxContribution`).
- Duración del proyecto en días, calculada a partir de las fechas de inicio y finalización.
- Número de países participantes, estimado a partir de los países presentes en `countryContr`.
- País coordinador (`coordinatorCountry`).

Durante el análisis inicial se identificaron 11 proyectos sin información suficiente para calcular su duración. Estos valores se sustituyeron por la mediana obtenida exclusivamente del conjunto de entrenamiento.

Las cuatro características numéricas se sometieron a una transformación logarítmica mediante `log1p` y posteriormente se estandarizaron utilizando `StandardScaler`.

Por su parte, el país coordinador se transformó mediante One-Hot Encoding. Como resultado, cada proyecto quedó representado mediante 57 características administrativas.

### 5.3. Modelos utilizados

En primer lugar, se entrenó un SVM lineal utilizando únicamente los metadatos. Se probaron configuraciones con pesos de clase balanceados y sin balancear. La segunda obtuvo mejores resultados en validación y fue seleccionada para la evaluación final.

A continuación, se recuperó el modelo TF-IDF + SVM desarrollado anteriormente para utilizarlo como referencia de clasificación basada exclusivamente en texto.

Finalmente, se creó una representación combinada concatenando las 50.000 características TF-IDF con las 57 características administrativas.

Para estudiar la influencia de los metadatos, se probaron tres ponderaciones: 1,0; 0,1 y 0,01. La selección se realizó utilizando el F1-macro de validación, obteniéndose el mejor resultado entre los modelos combinados con un peso de 0,01.

### 5.4. Resultados experimentales

Los resultados de los tres modelos seleccionados fueron los siguientes:

| Modelo | Accuracy validación | F1-macro validación | Accuracy prueba | F1-macro prueba |
|---|---:|---:|---:|---:|
| Solo texto | 0,5371 | 0,4127 | 0,5281 | 0,4072 |
| Solo metadatos | 0,1466 | 0,0223 | 0,1471 | 0,0214 |
| Texto + metadatos | 0,5365 | 0,4123 | 0,5276 | 0,4069 |

Los resultados muestran que el texto proporciona mucha más información útil para clasificar los proyectos que los metadatos administrativos.

El modelo basado únicamente en metadatos obtuvo un F1-macro de 0,0214 en prueba, mientras que el basado en texto alcanzó 0,4072.

Al combinar ambas fuentes de información, se obtuvo un F1-macro de 0,4069, prácticamente idéntico al del modelo basado únicamente en texto.

### 5.5. Análisis de las diferencias

Para estudiar el efecto de incorporar metadatos se compararon las predicciones de los modelos de texto y combinado sobre los 6.397 proyectos de prueba.

El modelo de texto clasificó correctamente 3.378 proyectos, mientras que el modelo combinado acertó 3.375.

Al incorporar los metadatos, 7 proyectos que antes se clasificaban incorrectamente pasaron a clasificarse correctamente. Sin embargo, otros 10 proyectos que antes se clasificaban correctamente pasaron a clasificarse incorrectamente.

También se realizó un análisis bootstrap pareado con 1.000 repeticiones.

La diferencia de F1-macro entre el modelo combinado y el modelo de texto fue de −0,000303, con un intervalo bootstrap del 95 % comprendido entre −0,001277 y 0,000752.

Como este intervalo incluye el valor cero, no se dispone de evidencia estadística suficiente para afirmar que añadir los metadatos produzca una mejora o un empeoramiento consistente del F1-macro.

### 5.6. Conclusiones de la extensión

Los experimentos realizados indican que el contenido textual de los resúmenes resulta mucho más útil para identificar la temática científica de los proyectos que los metadatos administrativos considerados.

Aunque los metadatos permiten obtener cierta información sobre los proyectos, su capacidad para distinguir entre las 39 categorías científicas es limitada.

Además, su incorporación al modelo TF-IDF + SVM no produjo una mejora apreciable del rendimiento.

Por tanto, para este conjunto de datos y las configuraciones evaluadas, el modelo basado únicamente en texto ofrece resultados equivalentes o ligeramente superiores a los del modelo combinado, sin necesidad de incorporar información administrativa adicional.

## 6. Conclusiones generales y limitaciones

### 6.1. Conclusiones generales

En este proyecto se han comparado diferentes técnicas de representación de texto y modelos de Machine Learning para clasificar proyectos de investigación del programa Horizonte 2020 según las categorías científicas de EuroSciVoc.

Los experimentos realizados muestran que TF-IDF combinado con un SVM lineal fue la alternativa que obtuvo el mejor F1-macro, alcanzando un valor de 0,4072 sobre el conjunto de prueba. Este resultado demuestra que los métodos tradicionales de representación de texto siguen siendo competitivos en problemas de clasificación documental.

Por otro lado, DistilBERT con fine-tuning consiguió el mayor accuracy, con un 55,56 % de aciertos. Sin embargo, su F1-macro fue inferior al de TF-IDF, lo que refleja la importancia de utilizar diferentes métricas cuando las categorías presentan una distribución desigual.

También se comprobó que el rendimiento mejora al clasificar los proyectos en las seis categorías generales de EuroSciVoc en lugar de utilizar las 39 categorías específicas. Esto pone de manifiesto la mayor dificultad de distinguir entre áreas científicas con un nivel de detalle elevado.

El análisis de incertidumbre permitió estudiar la variabilidad de los resultados y comparar estadísticamente algunos de los modelos.

Finalmente, la extensión mostró que los metadatos administrativos tienen una capacidad predictiva limitada frente al contenido de los resúmenes. Además, su incorporación al modelo de texto no produjo una mejora apreciable en el rendimiento.

En conjunto, los resultados indican que la elección de la representación textual, el clasificador y la métrica de evaluación tiene una influencia importante en la calidad de la clasificación.

### 6.2. Limitaciones del trabajo

A pesar de los resultados obtenidos, existen algunas limitaciones que deben tenerse en cuenta.

En primer lugar, el conjunto de datos presenta un desequilibrio entre categorías. Algunas áreas científicas contienen muchos más proyectos que otras, lo que dificulta la clasificación de las clases menos representadas.

Además, algunos proyectos están asociados a varios códigos EuroSciVoc. Para formular el problema como una clasificación multiclase, se seleccionó una única categoría de segundo nivel por proyecto. Por tanto, no se representa completamente el carácter multidisciplinar de determinadas investigaciones.

Otra limitación es que los resúmenes pueden contener terminología similar en categorías científicas diferentes, lo que provoca confusiones entre áreas relacionadas.

En el caso de DistilBERT, los textos se limitaron a un máximo de 512 tokens. Esto implica que, cuando un resumen supera esa longitud, parte de su contenido puede quedar fuera de la representación utilizada por el modelo.

Por último, las conclusiones corresponden al conjunto de datos y a las configuraciones evaluadas. No puede garantizarse que el mismo orden de rendimiento se mantenga con otros conjuntos de documentos, modelos o estrategias de entrenamiento.

### 6.3. Posibles mejoras

Como continuación de este trabajo, podrían estudiarse técnicas específicas para tratar el desequilibrio entre categorías o modelos capaces de asignar varias etiquetas a un mismo proyecto.

También sería interesante analizar otras arquitecturas Transformer y diferentes estrategias de ajuste fino, así como estudiar representaciones que aprovechen directamente la estructura jerárquica de EuroSciVoc.

Estas alternativas permitirían profundizar en el problema y explorar posibles mejoras sobre los resultados obtenidos.
