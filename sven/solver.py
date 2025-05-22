import time
from sven.inductions import *

def update(
    blades, uInfty, timeStep, timeSimulation, innerIter, 
    deltaFlts, startTime, iterationVect):

    iterationTime = time.time()
    
    ###########################################################################
    # Initialize all inductions
    ###########################################################################
    for (iBlade, blade) in enumerate(blades):
        blade.inductionsFromWake[:, :] = 0.
        blade.inductionsAtNodes[:, :] = 0.
        blade.wakeNodesInductions[:, :, :] = 0.

    ###########################################################################
    # Calculates the attachment point of the very first filament row (or 
    # trailing edge position)
    ###########################################################################
    for blade in blades:
        blade.updateFirstWakeRow()
        nearWakeLength = blade.nearWakeLength

    ###########################################################################
    # "wakeFilamentsInductionsOnBladeOrWake" : compute the filaments' 
    #                                          induction on blade centers
    ###########################################################################
    t0 = time.time()
    if (nearWakeLength > 2):
        wakeFilamentsInductionsOnBladeOrWake(blades, deltaFlts, "blade")

    t1 = time.time()
    print('wakeInductionsOnBlade: ', t1 - t0)

    ###########################################################################
    # These have to be set back to zero before gamma bound convergence loop
    ###########################################################################
    blade.gammaShed = np.zeros_like(blade.gammaShed)
    blade.gammaTrail = np.zeros_like(blade.gammaTrail)
   
    t0 = time.time()
    biotTime = 0

    ###########################################################################
    # Convergence loop over gammaBound
    ###########################################################################

    bladesGammaBounds = []
    for i in range(len(blades)):
        bladesGammaBounds.append(0.)
    for i in range(innerIter):
        tb0 = time.time()
        #######################################################################
        #(1) "nearWakeInduction" : calculates induced velocities of bound 
        #                           filaments from one blade on another
        #(2) "estimateGammaBound": knowing all induced velocities on the 
        #                           blade -> calculate the blade's effective 
        #                           velocity, angle of attack, lift coefficient 
        #                           -> determine new bound circulation value.
        #(3) "updateSheds/updateTrails" : knowing new bound circulation -> shed 
        #                                 and trail circulations can be
        #                                 compute from Kelvin's theorem.
        #######################################################################
        nearWakeInducedVelocities = nearWakeInduction(blades, deltaFlts)
        tb1 = time.time()
        biotTime += tb1 - tb0

        iBlade = 0
        for (blade, bladeInducedVelocities) in zip(blades, nearWakeInducedVelocities):
            bladesGammaBounds[iBlade] = blade.estimateGammaBound(uInfty, bladeInducedVelocities)
            blade.updateSheds(bladesGammaBounds[iBlade])
            blade.updateTrails(bladesGammaBounds[iBlade])

            blade.gammaBound = bladesGammaBounds[iBlade]
            iBlade += 1

    ###########################################################################
    # Store bound circulation value after convergence: important for next tstep
    ###########################################################################
    for (iBlade, blade) in enumerate(blades):
        blade.storeOldGammaBound(bladesGammaBounds[iBlade])

    t1 = time.time()
    print('gammaBoundUpdate: ', t1 - t0, biotTime)

    ###########################################################################
    # Compute all inductions on wake elements : 
    #(1)"wakeFilamentsInductionsOnBladeOrWake" : inductions from wake filaments 
    #                                            on all other wake filaments
    #(2)"bladeInductionsOnWake"                : inductions from blades on 
    #                                            wake filaments
    ###########################################################################
    t0 = time.time()
    if (nearWakeLength > 2):
        wakeFilamentsInductionsOnBladeOrWake(blades, deltaFlts, "wake")

    t1 = time.time()
    print('wakeOnWake: ', t1 - t0)

    t0 = time.time()
    bladeInductionsOnWake(blades, deltaFlts)
    
    t1 = time.time()
    print('bladeOnWake: ', t1 - t0)

    ###########################################################################
    # Once all inductions are known, the induced wake velocity is used to 
    # advect vortex filaments in the wake.
    ###########################################################################
    t0 = time.time()

    if (nearWakeLength > 2):
        for blade in blades:
            blade.advectFilaments(uInfty, timeStep)
    t1 = time.time()
    print('advection: ', t1 - t0)

    ###########################################################################
    #(1)"spliceNearWake"            : trail and shed filaments from the 
    #                                 second to last row take values of sheds
    #                                 and trails from first to second to last 
    #                                 row.
    #(2)"updateFilamentCirculation" : first row of filaments take trail and shed 
    #                                 circulations values computed after 
    #                                 gammaBound convergence loop.
    ###########################################################################

    if (nearWakeLength > 2):
        for blade in blades:
            blade.spliceNearWake()
            blade.updateFilamentCirulations()

    print('Full iteration time: ', time.time() - iterationTime)
    iterationVect.append([time.time() - iterationTime, time.time()-startTime])

    return

