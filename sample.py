from functools import reduce

from anp.ANP import ANP
from anp.MatricaUsporedbi import MatricaUsporedbi
from anp.MatricaZavisnosti import MatricaZavisnosti
from generator.Generetor import Generator
from snap.SNAP import SNAP
brojKlastera = 1
brojKriterija = 4
gen = Generator(brojKlastera, brojKriterija)
gen.generateAllComparisonMatrices(writeToFile=False)
gen.generateDependancyMatrices(writeToFile=False)

anpList = []
snapList = []

# fh = open("rez_%d_klust_%d_krit.txt" % (brojKlastera, brojKriterija), "w")
# fh.write("komb_usporedbe;komb_zavisnosti;konvergira;tezi nuli;cezarova suma;"
#          "ANP_tezine;SNAP_tezine;Razlika;Postotak_razlika_manjih_od_0.1;"+ '\n')
# for usporedba in gen.matrices:
#     for zavisnost in gen.zmatrices:
#         anp = ANP(usporedba, zavisnost)
#         anp.simulate()
#         # anpList.append(anp)
#         snap = SNAP(usporedba, zavisnost)
#         snap.simulate()
#         # snapList.append(snap)
#         razlike = abs(anp.tezine - snap.tezine)
#         brojManjihOd01 = reduce(lambda x, y: x + (1 if y <= 0.1 else 0), razlike, 0)
#         fh.write("%s;%s;%d;%d;%d;%s;%s;%s;%d\n" %
#                  (usporedba.kombinacija, zavisnost.kombinacija,
#                  anp.supermatrica.konvergira, anp.supermatrica.teziNuli, anp.supermatrica.cezarova,
#                   anp.tezine, snap.tezine, razlike, (brojManjihOd01/len(snap.tezine))*100))
#         # S.calculateLimitMatrix()
#         # print(S.L)
# fh.close()
print("Broj kombinacija za %d klastera i %d kriterija: %d" % (brojKlastera, brojKriterija, len(anpList)) )

U = MatricaUsporedbi(Generator.izradiMatricu(None, [1, 1, 1, 1, 1, 2], [1, 1, 1, 1, 1, 0.5], 4), [1, 1, 1, 1, 1, 2])
Z = MatricaZavisnosti(Generator.izradiMatricu(None, [0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1], 4, 0), [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
anp = ANP(U, Z)
anp.simulate()
snap = SNAP(U, Z)
snap.simulate()
print("Supermatrica\n", anp.supermatrica.S)
print("Supermatrica - granicna\n", anp.supermatrica.L)
print("CO", snap.CO)
print("CI", snap.CI)
print("CO - CI", snap.razlike)
print("Norma1", snap.norm)
print("Norma2", snap.norm2)
print("Tezine", snap.tezine)

