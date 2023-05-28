#PRIMA PARTE:
#INPUT: array n x n ordinati
#output:: grafico con comparazione tra dct2 fatto in pseudocodice da noi vs quello della libreria
#al variare di n il tempo impiegato e l'esecuzione dei 2 algoritmi

#SECONDA PARTE
#interfaccia che fa scegliere un img bmp
#input: img bmp, intero F ( ampiezza delle finestrelle dove si effettuerà la dct2), intero d tra 0,e (2F-2) -> soglia di taglio delle freq
#procedura: dividere in blocchi quadrati di f pixel fxf
        #per ogni blocco applicare la dct2 (libreria)
        #eliminare le frequenze k + l>=d
