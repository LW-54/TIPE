def premiers(n : int) -> list[int]:
    if n < 2 : 
        return []
    Pn = [2]
    for i in range(3, n+1, 2):
        j = 0
        pi_i = len(Pn)
        while j < pi_i and Pn[j] <= i**(1/2) and i%Pn[j]!=0 :
            j += 1
        if j == pi_i or Pn[j] > i**(1/2) :
            Pn.append(i)
    return Pn

