# Import the modules needed to run the script.
# https://www.bggofurther.com/2015/01/create-an-interactive-command-line-menu-using-python/
import os
import sys
import numpy as np
from anp.ANP import ANP
from snap.SNAP import SNAP
from generator.Generetor import Generator
from anp.MatricaUsporedbi import MatricaUsporedbi
from anp.MatricaZavisnosti import MatricaZavisnosti

# =======================
#     MENUS FUNCTIONS
# =======================

brojKlastera = 0
brojKriterija = 0


# Main menu
def main_menu():
    os.system('clear')
    os.system('cls')

    print("Opcije:")
    print("1 - Generiranje kombinacija usporedbi")
    print("2 - Generiranje kombinacija zavisnosti")
    print("3 - Testiraj ANP1 i SNAP")
    print("4 - Testiraj ANP2 i SNAP")
    print("5 - Testiraj ANP3 i SNAP")
    print("6 - Testiraj ANP4 i SNAP")
    print("i - unos matrica")
    print("0 - Kraj\n")
    choice = input(" >>  ")
    exec_menu(choice)

    return


def exec_menu(choice):
    os.system('clear')
    ch = choice.lower()
    if ch == '':
        menu_actions['main_menu']()
    else:
        try:
            menu_actions[ch]()
        except KeyError:
            print("Nepoznata opcija\n")
            menu_actions['main_menu']()
    return


# Menu 1
def generiraj_usporedbe():
    global brojKlastera
    global brojKriterija
    print("Generiranje kombinacija usporedbi!\n")

    if brojKlastera == 0:
        brojKlastera = input("Broj klastera: ")
    if brojKriterija == 0:
        brojKriterija = input("Broj kriterija: ")

    global gen
    if gen is None:
        gen = Generator(brojKlastera, brojKriterija)
    gen.generateAllComparisonMatrices(writeToFile=input("Pisi datoteku (1/0)"))
    return option_end()


# Menu 2
def generiraj_zavisnosti():
    global brojKlastera
    global brojKriterija
    print("Generiranje kombinacija zavisnosti!\n")

    if brojKlastera == 0:
        brojKlastera = input("Broj klastera: ")
    if brojKriterija == 0:
        brojKriterija = input("Broj kriterija: ")

    global gen
    if gen is None:
        gen = Generator(brojKlastera, brojKriterija)
    gen.generateDependancyMatrices(writeToFile=input("Pisi datoteku (1/0)"))
    return back()


# Menu 3
def testiraj_anp1():
    print("ANP 1 - normalizacija zbrojem bez fiktivne alternative")
    global U, Z
    checkDependencies()
    global anp
    anp = ANP(U, Z)
    anp.simulate(matricaPrijelaza=False)
    return option_end()

# Menu 4
def testiraj_anp2():
    print("ANP 1 - normalizacija zbrojem s fiktivnom alternativom")
    global U, Z
    checkDependencies()
    global anp
    anp = ANP(U, Z)
    anp.simulate(matricaPrijelaza=True, fiktivnaAlt=True)
    return option_end()


# Menu 5
def testiraj_anp3():
    print("ANP 1 - normalizacija matricom prijelaza bez fiktivne alternative")
    global U, Z
    checkDependencies()
    global anp
    anp = ANP(U, Z)
    anp.simulate(matricaPrijelaza=True)
    return option_end()


# Menu 6
def testiraj_anp4():
    print("ANP 1 - normalizacija matricom prijelaza s fiktivnom alternativom")
    global U, Z
    checkDependencies()
    global anp
    anp = ANP(U, Z)
    anp.simulate(matricaPrijelaza=True, fiktivnaAlt=True)
    return option_end()


def inputMatrices():
    os.system('clear')
    os.system('cls')
    global brojKlastera
    global brojKriterija
    global U, Z
    load_defaults = input("Učitaj predefinirane matrice s 1 klasterom i 4 kriterija? (y/n)")
    if load_defaults.lower() == 'y':
        U = MatricaUsporedbi(Generator.izradiMatricu([2, 2, 2, 1, 1, 1], [1/2, 1/2, 1/2, 1, 1, 1], 4))
        Z = MatricaZavisnosti(Generator.izradiMatricu([2, 3, 4, 1, 1, 3], [2, 1, 4, 2, 1, 2], 4, 0))
    else:
        brojKlastera = int(input("Broj klastera: "))
        brojKriterija = int(input("Broj kriterija: "))
        br = int((brojKriterija - 1) * brojKriterija / 2)
        ulaz = input("Gornji trokut matrice usporedbe (%d elemenata): " % br)
        usporedba = np.array(list(map(int, ulaz.split(","))))
        U = MatricaUsporedbi(Generator.izradiMatricu(usporedba, 1 / usporedba, brojKriterija))
        print(U.U)
        ulazZ1 = input("Gornji trokut matrice zavisnosti (%d elemenata): " % br)
        ulazZ2 = input("Donji trokut matrice zavisnosti (%d elemenata): " % br)
        gornji = np.array(list(map(int, ulazZ1.split(","))))
        donji = np.array(list(map(int, ulazZ2.split(","))))
        Z = MatricaZavisnosti(Generator.izradiMatricu(gornji, donji, brojKriterija))
        print(Z.Z)
    global snap
    snap = SNAP(U, Z)
    return back()

def printResults():
    global anp, snap
    anp.printResults()
    snap.simulate()
    snap.printResults()
    return back()

# Return to main menu
def back():
    menu_actions['main_menu']()


# Give option to return to main menu
def option_end():
    print("9. Glavni meni")
    print("0. Kraj programa")
    print("p - ispisi rezultate")
    choice = input(" >>  ")
    exec_menu(choice)
    return

# Exit program
def done():
    sys.exit()

def checkDependencies():
    global U, Z
    if U == None:
        generiraj_usporedbe()
    if Z == None:
        generiraj_zavisnosti()


# =======================
#    MENUS DEFINITIONS
# =======================

# Menu definition
menu_actions = {
    'main_menu': main_menu,
    '1': generiraj_usporedbe,
    '2': generiraj_zavisnosti,
    '3': testiraj_anp1,
    '4': testiraj_anp2,
    '5': testiraj_anp3,
    '6': testiraj_anp4,
    '9': back,
    '0': done,
    'i': inputMatrices,
    'p': printResults
}

if __name__ == "__main__":
    main_menu()
