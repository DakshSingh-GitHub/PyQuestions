def add_item(item, container=[]):
    container.append(item)
    return container

l = [[1, 2], [3, 4]]
l2 = l.copy()

l[0].append(99)

print(l)
print(l2)