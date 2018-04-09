# Import the modules needed to run the script.
# https://www.bggofurther.com/2015/01/create-an-interactive-command-line-menu-using-python/
import os
import sys
from generator.Generetor import Generator

# =======================
#     MENUS FUNCTIONS
# =======================

brojKlastera = 0
brojKriterija = 0


# Main menu
def main_menu():
    os.system('clear')

    print("Opcije:")
    print("1 - Generiranje kombinacija usporedbi")
    print("2 - Generiranje kombinacija zavisnosti")
    print("3 - Testiraj ANP1 i SNAP")
    print("4 - Testiraj ANP2 i SNAP")
    print("5 - Testiraj ANP3 i SNAP")
    print("6 - Testiraj ANP4 i SNAP")
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
    print("9. Glavni meni\n")
    print("0. Kraj programa\n")
    if brojKlastera == 0:
        brojKlastera = input("Broj klastera: ")
    if brojKriterija == 0:
        brojKriterija = input("Broj kriterija: ")

    global gen
    if gen is None:
        gen = Generator(brojKlastera, brojKriterija)
    gen.generateAllComparisonMatrices(writeToFile=input("Pisi datoteku (1/0)"))

    choice = input(" >>  ")
    exec_menu(choice)
    return


# Menu 2
def generiraj_zavisnosti():
    global brojKlastera
    global brojKriterija
    print("Generiranje kombinacija zavisnosti!\n")
    print("9. Glavni meni\n")
    print("0. Kraj programa\n")

    if brojKlastera == 0:
        brojKlastera = input("Broj klastera: ")
    if brojKriterija == 0:
        brojKriterija = input("Broj kriterija: ")

    global gen
    if gen is None:
        gen = Generator(brojKlastera, brojKriterija)
    gen.generateDependancyMatrices(writeToFile=input("Pisi datoteku (1/0)"))
    choice = input(" >>  ")
    exec_menu(choice)
    return


# Glavni meni to main menu
def back():
    menu_actions['main_menu']()


# Exit program
def done():
    sys.exit()


# =======================
#    MENUS DEFINITIONS
# =======================

# Menu definition
menu_actions = {
    'main_menu': main_menu,
    '1': generiraj_usporedbe,
    '2': generiraj_zavisnosti,
    '9': back,
    '0': done,
}

if __name__ == "__main__":
    main_menu()
