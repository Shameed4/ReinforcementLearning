def promptTwoLetters(prompt, l1, l2):
    l1, l2 = l1.strip().lower()[0], l2.strip().lower()[0]
    inp = input(f'{prompt}: ({l1[0]}|{l2[0]}): ').strip().lower()[0]
    while inp != l1 and inp != l2:
        print(f'Invalid input. Please enter {l1} or {l2}')
        inp = input(f'{prompt}: ({l1[0]}|{l2[0]}): ').strip().lower()[0]
    return inp == l1


def promptYesNo(prompt):
    return promptTwoLetters(prompt, 'y', 'n')


def promptDigit(prompt):
    prompt += ": "
    inp = input(prompt)
    while not inp.isdigit():
        print("Please enter a valid integer.")
        inp = input(prompt)
    return int(inp)


def promptFloat(prompt):
    prompt += ": "
    inp = input(prompt)
    while True:
        try:
            float(inp)
            return float(inp)
        except ValueError:
            inp = input(inp)


if __name__ == "__main__":
    while True:
        print(promptFloat("Enter a float"))