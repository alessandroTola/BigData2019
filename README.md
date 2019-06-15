# PROGETTO BIG DATA 2019
Progetto bigdata natural language processing, analizzare e classificare recensioni di prodotti amazon

## Dataset
Abbiamo utilizzato un dataset contenente le review di Amazon, che Ë possibile scaricare (su richiesta), [a questo indirizzo](http://jmcauley.ucsd.edu/data/amazon/links.html). Il dataset Ë in formato .json.

## File dei word embeddings
I word embeddings che abbiamo utilizzato sono reperibili [a questo indirizzo](http://www.maurodragoni.com/research/opinionmining/dranziera/embeddings-evaluation.php)

## Preprocessing del dataset
La creazione del dataset prevede le seguenti fasi:

-creazione di una colonna 'reviewTS' con la concatenazione del titolo della recensione e del suo contenuto
-eliminazione di tutte le colonne non utili come, 'reviewerID' o 'unixReviewTime'.
-eliminazione delle recensioni con 3 stelle, per avere una suddivisione tra recensioni positive(4-5 stelle) e negative (1-2 stelle)
-inserimento della label per le istanze positive e negative
-eliminazione delle colonne non pi˘ utili, come stelle, testo della recensione e titolo
-suddivisione del dataset in due parti, il training set costituito dall'80% dei dati e il test set  costituito dal restante 20%
-suddivisione dell testo presente nella colonna 'reviewTS' in singole parole
-rimozione delle parole poco influenti come per esempio congiunzioni e articoli
-calcolo delle features a partire dalle parole separate e filtrate, ottenuto andando a cercare ogni parola tra i word embeddings
-precalcolo e somma dei valori tra loro per ogni parola della recensione
-eliminazione di tutte le colonne escluse quelle con 'label' e 'features'

A questo punto i file sono pronti per la fase di addestramento

## Modello addestrato utilizzando word embeddings
La pipeline prevede due passaggi:
1. Pulizia e creazione del dataset, ottenute eseguendo il file ```clean_json.sh```. Al suo interno Ë necessario inserire path e nome sia del file .json di partenza sia del file contenente i word embeddings precalcolati (in formato csv). Questo procedimento Ë stato eseguito in locale per evitare sovraccarichi sulla RAM nel cluster. L'output consister‡ nel training set e nel test set con il giusto formato per l'addestramento.
2. Addestramento del modello per mezzo del file ```modelEmb.sh```. Eseguiamo una logistic regression seguita da una crossvalidation ten-fold. I risultati del training e del test set vengono poi esportati nel file 'results.txt'.
