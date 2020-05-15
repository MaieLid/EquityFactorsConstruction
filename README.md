# EquityFactorsConstruction
A replication of the works of Fays, Hübner and Lambert (2018) analysing the Fama French factors construction


### Données nécessaires (de juin 1963 à décembre 2019) :

2 fichiers de données de fama french:
- fama_french_ts.pkl = facteurs de fama french 
- 100_Portfolios_10x10.csv = composition des 100 portefeuilles size/value de Fama French

6 fichiers calculés par nos soins pour accélérer une réutilisation :
- (all_exch/NYSE)\_(sz/bm)\_(2/3)x3.pkl = breakpoints à utliser pour les tris

5 fichiers téléchargés de WRDS :
- crsp_m.pkl = base crsp (information financières supplémentaires)
- comp.pkl et comp_with_lt.pkl = base compustat (base de données de S&P contenant des data fondamentales et marché sur des stocks US et canada)
- ccm.pkl = univers américain
- dlret.pkl = delisting returns
