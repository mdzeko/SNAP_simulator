from anp.ANP import ANP
from anp.MatricaUsporedbi import MatricaUsporedbi
from anp.MatricaZavisnosti import MatricaZavisnosti
from generator.Generetor import Generator
from snap.SNAP import SNAP

gen = Generator(1, 4)
gen.generateAllComparisonMatrices()
gen.generateDependancyMatrices()



anpList = []
snapList = []

# for usporedba in gen.matrices:
#    for zavisnost in gen.zmatrices:
#        anp = ANP(usporedba, zavisnost)
#        anp.simulate()
#        anpList.append(anp)
#        snap = SNAP(usporedba, zavisnost)
#        snap.simulate()
#        snapList.append(snap)
        # S.calculateLimitMatrix()
        # print(S.L)


U = MatricaUsporedbi(Generator.izradiMatricu(None, [2, 2, 2, 1, 1, 1], [0.5, 0.5, 0.5, 1, 1, 1], 4))
Z = MatricaZavisnosti(Generator.izradiMatricu(None, [2, 3, 4, 1, 3, 3], [2, 1, 4, 2, 1, 2], 4, 0))
snap = SNAP(U, Z)
snap.simulate()
print("CO", snap.CO)
print("CI", snap.CI)
print("CO - CI", snap.razlike)
print("Norma1", snap.norm)
print("Norma2", snap.norm2)
print(snap.tezine)