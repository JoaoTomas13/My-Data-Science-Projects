Title: Data Fusion - Problem – Life Insurance

Problem: Individuals with suspected COVID are admitted to the
hospital emergency room
▪ At the time of admission, several variables/parameters are
acquired (low cost and simple to acquire)
▪ Based on these variables, the health professional must
decide whether the individual remains hospitalized for
additional examinations or should return home.

Dataset: 
▪x1 Gender {0,1} = { Female, Male}
▪ X2 Age [34 .. 99]
▪ X3 Marital status {0,1} = { single, married}
▪ X4 Vaccinated {0,1} = { No, Yes}
▪ X5 Breathing difficulty {0,1,2,3} = { none, some, moderate, high}
▪ X6 Heart Rate [38.. 272]
▪ X7 Blood pressure [115.. 164]
▪ X8 Temperature [36.00 .. 38.98]
▪ X9 Clinical Guidelines A rule based on the breathing difficulty and the temperature
▪ T Decision Final decision
{0,1} = { return home, stay at hospital }

Questions to answer: 
▪ Is the performance of the classifier acceptable ?
▪ Should all information (inputs/variables) be used ?
▪ Discrete versus continuous variables ?
▪ Conditional probabilities : normal distribution ?

-The notebook is majority in portuguese-