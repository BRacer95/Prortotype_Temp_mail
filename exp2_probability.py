import random

trials = 10000
even_count = sum(1 for _ in range(trials) if random.randint(1, 6) % 2 == 0)
print(f"Experimental probability of even number: {even_count / trials}")