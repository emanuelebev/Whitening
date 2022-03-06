# Whitening
La repository in questione fa riferimento alla relazione di tirocinio: Algoritmi di Whitening

Sono presenti 3 algoritmi di whitening utilizzati per i test:
- whintening_eigendecomp.py: algortimo di whitening che calcola autovalori e autovettori usando la eigendecomposition
- whitening_SVD.py: algortimo di whitening che calcola autovalori e autovettori usando SVD (Singular Value Decomposition)
- whitening_Cholesky.py: algortimo di whitening usando la decomposizione di Cholesky

Tutte le altre cartelle contengono matrici di dati json utilizzati come imput dal programma e per effettuare test

Cartella algoritmi CVS contiene gli algoritmi utilizzati per i test ma modificati in modo da leggere file cvs in input:
-cvs_eig.py: algoritmo di whitening con eigendecomposition
-csv_svd.py: algoritmo di whitening con SVD

Cartella SVD alternative whitening contiene un algoritmo di whitening alternativo a quello della cartella test, implementato sia per file in input di tipo CSV che json.

Cartella Multi whitening scelta contiene un algoritmo di whitening che implementa i 3 metodi di whitening, ovvero ZCA, PCA e Cholesky in maniera rapida ma poco specifica. Implementato sia per file in input di tipo json e CSV.

Cartella Basic Whitening 
