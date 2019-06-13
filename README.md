# PROGETTO BIG DATA 2019
Progetto bigdata natural language processing, analizzare e classificare recensioni di prodotti amazon

## Dataset
Abbiamo utilizzato il dataset che è possibile scaricare (su richiesta), [qui](http://jmcauley.ucsd.edu/data/amazon/links.html). Il dataset è in formato json.

## File dei word embedding
[qui](http://www.maurodragoni.com/research/opinionmining/dranziera/embeddings-evaluation.php)

## Preprocessing del dataset
La creazione del dataset prevede le seguenti fasi:

-creazione di una colonna 'reviewTS' con la concatenazione del titolo della recensione e del suo contenuto
-eliminazione di tutte le colonne non utili come, 'reviewerID' o 'unixReviewTime'.
-eliminazione delle recensioni con 3 stelle, per vere una suddivisione tra positivi (4-5 stelle) e negative (1-2 stelle)
-inserita la label per le istanze positive e negative
-eliminate lo colonne non più utili, come stelle, testo recensione e titolo
-l'intero dataset viene suddiviso in due parti, trainingset 80% e testset 20%
-la colonna il testo presente nella 'reviewTS' viene suddiviso in singole parole
-rimozione delle parole poco influenti come per esempio congiunzioni articoli
-utilizzando le parole separate e filtrate vengono calcolate le feature andando a cercare ogni parola tra i word embeddings
precalcolati e sommati tra loro per ogni parola della recensione
-eliminazione di tutte le colonne escluse quelle con 'laber' e 'features'

A questo punto i file sono pronti per la fase di addestramento

## Modello addestrato utilizzando word embedding
La pipeline prevede due passaggi, la parte di pulizia e creazione del dataset eseguendo il file ```clean_json.sh```. Al suo interno è necessario inserire il path e nome sia del file json di partenza che i file con i word embeddings precalcolati (in formato csv). Questo procedimento è stato eseguito in locale per evitare sovracarichi di Ram nel cluster.
L'output sarà il trainigset e testset con il giusto formato per l'addestramento, 'label' e 'features'
Eseguendo il secondo file ```modelEmb.sh``` andremo ad addestrare il modello, eseguendo una crossvalidation con 10 fold, al suo interno vi è la scrittura di un file 'results.txt' con tutti i risultati del training e del test.
