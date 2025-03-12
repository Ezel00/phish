import random

p1 = r"C:\Users\ezele\Desktop\thesis\tdaPython\final\notPhish.txt"
p2 = r"C:\Users\ezele\Desktop\thesis\tdaPython\final\phish.txt"
def readF(file_path):
    with open(file_path,  'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]

# divides the lists into test and train, verifies they dont have anything in common
def checks(p2, p1):    
    phishL = readF(p2)
    notPhish = readF(p1)
    def randomPick(original_list, num_random_pick):
        """Randomly picks elements from a list and removes them from the original list."""
        if len(original_list) < num_random_pick:
            print(f"Warning: list has fewer than {num_random_pick} elements, picking all elements.")
            picked_elements = original_list
            original_list.clear()  
        else:
            picked_elements = random.sample(original_list, num_random_pick)
            for item in picked_elements:
                original_list.remove(item)  
        return picked_elements, original_list
    
    picked, rest = (randomPick(phishL,3000))
    picked2, rest2 = (randomPick(notPhish,3000))
    #checks that there are no common items
    def compare(list1, list2):
        common_elements = set(list1).intersection(set(list2))
        print(f"Common elements between list1 and list2: {common_elements}")
        return 0
    
    compare(picked, rest)
    compare(picked2, rest2)
    compare(picked, rest2)
    
    n1 = r"C:\Users\ezele\Desktop\thesis\tdaPython\final\phishTrain.txt"
    n2 = r"C:\Users\ezele\Desktop\thesis\tdaPython\final\phishTest.txt"
    n3 = r"C:\Users\ezele\Desktop\thesis\tdaPython\final\notpTrain.txt"
    n4 = r"C:\Users\ezele\Desktop\thesis\tdaPython\final\notpTest.txt"
    
    def saveToTxt(file_name, list_data):
        with open(file_name, 'w', encoding='utf-8') as file:
            for item in list_data:
                file.write(f"{item}\n")
        print(f"List saved to {file_name}")
    
    saveToTxt(n1, rest)
    saveToTxt(n2, picked)
    saveToTxt(n3, rest2)
    saveToTxt(n4, picked2)
