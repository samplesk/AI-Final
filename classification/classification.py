#learning from 18 attributes (animal name, 15 Boolean, 2 numerics)
#The final number is the set it falls into, 1 through 7

zoo.dataloader = torch.utils.DataLoader(dataset, batch_size=1, suffle=False)
 
#loop for lines in set
#loop for each attribute
#define each seperate?
    #name
    #hair
    #feathers
    #lays eggs
    #makes milk
    #can fly
    #is aquatic
    #.....
#compile and decide which of 7 categories it goes in