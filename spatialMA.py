import numpy as np
import random
import matplotlib.pyplot as plt
import csv 

class cell:
    def __init__(self, kind, coords):
        # Marking for display method
        self.kind = kind
        self.coords = coords
        self.radius = 0
        # Selective effect
        self.s = float(0)
        self.fitness = 1
    
    def getFitness(self):
        return self.fitness
    def setFitness(self, fitness):
        self.fitness = fitness
    def getKind(self):
        return self.kind
    def setKind(self, kind):
        self.kind = kind
    def getCoords(self):
        return self.coords
    def getRad(self):
        return self.radius
    def setRad(self, rad):
        self.radius = rad
    def getS(self):
        return self.s
    def setS(self, s):
        self.s = float(s)
    


class plate:
    
    # Plate initialization
    def __init__(self, size):
        self.size = size
        self.grid = np.empty(shape=(size,size), dtype=object)
        self.MAX_RADIUS = int((size-2)/2)
        self.mutantList = []
        self.center = int(size/2)
        # Set that will contain all individuals on the current colony front
        self.edgeCoords = set()
        # Radius of the circular mask
        self.radius = 1
        self.dt = 0.1
        # Total time elapsed. Set to the increment at first
        self.T = self.dt
        
        # Initialize the plate with all the cells
        for i in range(size):
            for j in range(size):
                self.grid[i,j] = cell(0,[i,j])

        # Find the centre of the plate and set the seeding indiviual (WT) to 
        # have a 'kind' of 1. Its selective effect is zero.
        centre = int((size-1)/2)
        self.centreCell = self.grid[centre, centre]
        self.centreCell.setKind(1)
        # Ogrid creates an x,y coordinate framework within a 2d array 
        self.y,self.x = np.ogrid[-centre: centre+1, -centre: centre+1]
        self.ansS = float(0)


    def getMutants(self):
        return self.mutantList

   
   # Method to grow the next generation.
    def nextGen(self, mu = 0.00043):
        dr = self.dt * (self.centreCell.getFitness())

        # Circular mask created to grow the current population by dr
        mask = self.x**2+self.y**2 <= self.radius**2
        nGen = 1*mask.astype(float)
        mask = self.x**2+self.y**2 <= (self.radius-dr)**2
        pGen = 1*mask.astype(float)
        
        indices = np.where((nGen - pGen) == 1)
        outerLayer = zip(*indices)
        outerGenCoords= list(outerLayer)
        
        # For each cell that lay on the next generation's colony front... we seek to occupy it.
        # But first make sure its empty, then occupy it by the WT
        for outerCellCoords in outerGenCoords:
            outerCell = self.grid[outerCellCoords[0], outerCellCoords[1]]
            if outerCell.getKind() == 0:
                outerCell.setKind(1)
                # We add these newly "grown" cells to a list of possible coordinates
                # that a mutation can take place. i.e. truly on the edge 
                self.edgeCoords.add(tuple(outerCell.getCoords()))

        # Calculating the number of possible mutants in the next generation
        numMut = int(np.random.poisson(lam=len(self.edgeCoords)*mu, size=1)[0])
        
        if numMut > 0:
            # We sample numMut cells from the list of cells in which a mutation can place (on the edge)
            mutantCoords = random.sample(self.edgeCoords, numMut)
            
            for mutantCoord in mutantCoords:
                mutantCell = self.grid[mutantCoord[0], mutantCoord[1]]
                mutantCell.setKind(1.5)
                mutantCell.setRad(1)

                # The selective effects are drawn from the following distribution
                s = np.random.normal(0,0.75,1)[0]
            
                # If the selective effect is less than negative one, the fitness of the individual no longer
                # makes sense (since speed of growth is our fitness criteria - negative growth does not 
                # make sense.)
                if s < -1:
                    s = -1
                
                # Update the selective effect and fitness values
                mutantCell.setS(s) 
                newFit = mutantCell.getFitness() * (1+s)
                mutantCell.setFitness(newFit)
                
                self.mutantList.append(mutantCell.getCoords())

        # After the procedure is done, we incremenent the radius, clear the list of cells
        # on the colony front and incremenet the total time elapsed
        self.radius = self.radius + dr
        self.edgeCoords.clear()
        self.T += self.dt


    # Method to grow mutants
    def growMutants(self):
        # Variable needed increment each mutant marking for the display mutants. This way
        # each mutant is shown in a different color
        base = 0
        # Loop over the mutant ancestors
        for mutantCoords in self.mutantList:
            xm = mutantCoords[0]
            ym = mutantCoords[1]
            # Find the cell itself
            mutantCell = self.grid[xm,ym]
            r = mutantCell.getRad()
            fitness = mutantCell.getFitness()
            dr = self.dt * fitness

            # Expand the mutation at the given mutant speed
            mask = (self.x - (ym-self.center))**2 + (self.y - (xm-self.center))**2 <= r**2
            nGen = 1*mask.astype(float)
            mask = self.x**2+self.y**2 <= (self.radius-dr)**2
            prev = 1*mask.astype(float)
            indices = np.where((nGen-prev) == 1)
            # Coordinates of all the different mutant cell in the next generation
            # (i.e. the centre points for each mutant growth mask)
            mutatedCoords = list(zip(*indices))

            for coords in mutatedCoords:
                chosenCell = self.grid[coords]
                # We grow these mutants IF the cell is empty
                if chosenCell.getKind() == 0:
                    chosenCell.setKind(2 + base)
                    chosenCell.setS(mutantCell.getS())
                    # Add these pixels to the expanding edge choices for new mutations 
                    # to potential arrise from
                    self.edgeCoords.add(tuple(chosenCell.getCoords()))

            base = base + 1
            mutantCell.setRad(r + dr)


    # Method that helps in creating the DFE. In specific this method curates a list of all
    # individuals that have non-zero selective effect, then randomly selects one. 
    def dfe(self):
        allS =[]
        for i in range(self.size):
            for j in range(self.size):
                extractedCell = self.grid[i,j]
                extractedS = extractedCell.getS()
                
                if extractedS != 0:
                    allS.append(extractedS)
        # We want to make sure we have selected an individual, otherwise we return a flag to 
        # the calling command
        if len(allS) > 0:
            return random.choice(allS) 
        else:
            return -999


    # Method to create data that estimates the population's growth rate
    def growthRateEstimate(self):
        numIndiv = 0
        displayGrid = np.zeros((self.size,self.size))
        
        for i in range(self.size):
            for j in range(self.size):
                extractedCell = self.grid[i,j]
                extractedKind = extractedCell.getKind()
                # Each time we find an cell that isn't empty we add an individual to the counter
                if extractedKind != 0:
                    numIndiv += 1
        return self.T, numIndiv
            

    # Display method
    def display(self, frame=-1, save = False):
        # Create the matrix to be displayed
        displayGrid = np.zeros((self.size,self.size))
        # Loop over the plate cells
        for i in range(self.size):
            for j in range(self.size):
                extractedCell = self.grid[i,j]
                extractedKind = extractedCell.getKind()

                # Coding each entry of the display matrix to coincide with markings
                # Empty cell
                if extractedKind == 0:
                    displayGrid[i,j] = 0
                # WT
                elif extractedKind == 1:
                    displayGrid[i,j] = 1
                # Mutant
                else:
                    displayGrid[i,j] = extractedCell.getKind()
        
        # Create the figure and save the frame
        fig1, ax1 = plt.subplots() 
        ax1.matshow(displayGrid)
        ax1.axis('off')

        if save == False:
            plt.show()
        else:
            fig1.savefig('/Users/tjayoub/Desktop/Thesis/Python/images/ft-{0}.png'.format(frame))
            plt.close(fig1)

       

        
        
def createFrames(size, T, save):
    p = plate(size)
    for i in range(1, T):
        mu = 0.00043
        p.nextGen(mu)
        p.growMutants()
        p.display(i, save)


def createDFEdata(size, replicate, loop):
    fileName = "dfeData_{}rep_{}genLoop.txt".format(replicate, loop)
    with open(fileName, "a") as f:
        writer = csv.writer(f)
        
        for j in range(replicate):
            p = plate(size)
            for i in range(1, loop):
                mu = 0.00043
                p.nextGen(mu)
                p.growMutants()
            
            randomSelection = p.dfe()
            if randomSelection != -999:
                f.write(str(randomSelection) + "\n")
            else:
                pass

def createGrowthEstimateData(size, loop):
    fileName = "growthRateEstimates_{}genLoop.txt".format(loop)

    with open(fileName, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "numIndiv"])

        p = plate(size)
        for i in range(1, loop):
            mu = 0.00043
            p.nextGen(mu)
            p.growMutants()

            (dt, numIndiv) = p.growthRateEstimate()
            writer.writerow([dt, numIndiv])


#createFrames(331, 10, True)
#createDFEdata(301, 5, 700)
#createGrowthEstimateData(301, 300)
