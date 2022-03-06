# Whitening
La repository in questione fa riferimento alla relazione di tirocinio: Algoritmi di Whitening

Sono presenti 3 algoritmi di whitening utilizzati per i test:
- whintening_eigendecomp.py: algortimo di whitening che calcola autovalori e autovettori usando la eigendecomposition
- whitening_SVD.py: algortimo di whitening che calcola autovalori e autovettori usando SVD (Singular Value Decomposition)
- whitening_Cholesky.py: algortimo di whitening usando la decomposizione di Cholesky

Tutte le altre cartelle contengono matrici di dati json utilizzati come imput dal programma e per effettuare test

Cartella algoritmi CSV contiene gli algoritmi utilizzati per i test ma modificati in modo da leggere file cvs in input:
-cvs_eig.py: algoritmo di whitening con eigendecomposition
-csv_svd.py: algoritmo di whitening con SVD

Cartella SVD alternative whitening contiene un algoritmo di whitening alternativo a quello della cartella test, implementato sia per file in input di tipo CSV che json.

Cartella Multi whitening scelta contiene un algoritmo di whitening che implementa i 3 metodi di whitening, ovvero ZCA, PCA e Cholesky in maniera rapida ma poco specifica. Implementato sia per file in input di tipo json e CSV.

Cartella Basic Whitening contiene due algoritmi di whitening che simulano lo "sbiancamento" di punti in un asse cartesiano (in particolare 5 e 1000 punti), mostrando anche il risultato della distribuzione dei suddetti punti dopo aver effettuato i calcoli, in formato png utilizzando la libreria di python matplotlib. Queste immagini si trovano nella cartella plot img.

Cartella Verify contiene gli algoritmi per verificare la correttezza delle equazioni presenti nella parte teorica del whitening. In questo algoritmo viene utilizzata la matrice dim.json, 11 righe e 10 colonne. Tutte le matrici calcolate dall'algoritmo sono presenti nella cartella e descritte dal file
