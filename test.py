from anp.ANP import ANP
from anp.MatricaUsporedbi import MatricaUsporedbi
from anp.MatricaZavisnosti import MatricaZavisnosti
from generator.Generetor import Generator
from snap.SNAP import SNAP
import time

totalElapsed = time.process_time()
brojKlastera = 1
brojKriterija = 5


usp = MatricaUsporedbi(
    Generator.izradiMatUsporedbe([5.0, 5.0, 0.5, 0.5, 0.2, 0.16666666666666666, 1/5, 0.5, 0.5, 1], brojKriterija))
zav = MatricaZavisnosti(
    Generator.izradiMatZavisnosti([4, 1, 2, 2, 2, 2, 2, 3, 3, 3, 1, 1, 2, 4, 2, 2, 2, 3, 3, 3], brojKriterija))
snap = SNAP(usp.weights, zav.Z)
anp = ANP(usp.weights, zav.Z)
snap.simulate()
snap.printResults()


# anp = ANP(U, Z)
# anp.simulate()
# snap = SNAP(U, Z)
# snap.simulate()
# print("Supermatrica\n", anp.supermatrica.S)
# print("Supermatrica - gransicna\n", anp.supermatrica.L)
# print("CO", snap.CO)
# print("CI", snap.CI)
# print("CO - CI", snap.razlike)
# print("Norma1", snap.norm)
# print("Norma2", snap.norm2)
# print("Tezine", snap.tezine)
