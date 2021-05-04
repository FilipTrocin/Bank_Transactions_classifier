# Bank_Transactions_classifier

## Introduction

The aim of the project was to analyse the bank transactions and to classify them appropriately using Neural Networks. 

## Classification Model
1. Before delivering data to the model, the data was preprocessed
- Preprocessing consisted of removing unnecessary parts of the descriptions, which could affect the accuracy of the model
- (regex, replace method, or stopwords from nltk.corpus)

2. Dictionary of all unique words found in the given records was created, and subsequently, every transaction was converted to its digital representation (based on the index assigned in dictionary)
- (Tokenizer from keras preprocessing)

3. Lastly, to make every transaction the same size (5 in this case), to each one the sequence
of 0s were padded
- (pad_sequences from keras preprocessing)


The model is not ideal. Although its high loss score value – 12.470, the model has an acceptable accuracy amounting to 83.20% (best value I have managed to receive in this model architecture and those parameters. Model with that accuracy was saved to the file). There is no overfitting in that model, meaning the model performs similarly on the validation data, as on the training set. In order to enhance the accuracy of the model, changing parameters will probably not be sufficient and rebuilding the model architecture would be advisable

*Known issue: The model has difficulties with distinguishing records which should belong to TRAVEL category, however they are classified to the ACCOMODATION AND MEALS*
 

## Data Analysis


### Division of Data
- The given dataset consisted of 12 500 records
- 80% (10 000) out of them were dedicated for training purposes, whereas the remaining 20% (2500) for validating results

![alt text](https://github.com/FilipTrocin/Bank_Transactions_classifier/blob/main/graphs/graph5.png?raw=true)


### Rundown of the number of transactions and their amounts

- People mainly made low-budget purchases
- Transactions made for the amount greater than £40 constituted less than 10% of all transactions
- Transactions from the range £0-£10 and £20- £30 contributed the most, accounting for 31.1% and 26.42% respectively

 ![alt text](https://github.com/FilipTrocin/Bank_Transactions_classifier/blob/main/graphs/graph1.png?raw=true)


### Rundown of the number of transactions and their types

- Debit Card transactions were the most common forms of payment (37.7%), while Faster Payments Outwards were the rarest (5.4%)
- The second most frequently performed transactions were Charges and Cash with a difference of approximately 17% and 21% compared to the leading transaction
- Mobile Payment Outgoing and Direct Debit contributed similarly, with a difference of 3% between each other
 
  ![alt text](https://github.com/FilipTrocin/Bank_Transactions_classifier/blob/main/graphs/graph2.png?raw=true)


### Rundown of the number of transactions and their descriptions

- The forefront of the transactions comes from three different areas
- The most common banking transaction was travelodge, which represents the travel sector - over 300 transactions
- PayPal placed the second position, being behind the leading travelodge by approximately 30 transactions
 
 ![alt text](https://github.com/FilipTrocin/Bank_Transactions_classifier/blob/main/graphs/graph3.png?raw=true)
 
 
 ### Rundown of the number of transactions and their descriptions
 
 - The singular transactions came mainly from the "accommodation and meals" sector, where people were most likely going to eat out

 ![alt text](https://github.com/FilipTrocin/Bank_Transactions_classifier/blob/main/graphs/graph4.png?raw=true)

