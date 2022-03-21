# Functions for algorithms that necessitate back and forth between PyTorch actors and genotype

def get_dim_gen(actor):
    """ 
    Get the genotype number of dimesions of a DNN. 
    Inputs: actor {Actor} - the actor 
    Outputs: dim_gen {int} - the genotype dimension
    """
    state_dict = actor.state_dict()
    dim_gen = 0
    for tensor in state_dict:
      if "weight" or "bias" in tensor:
          if len(list(state_dict[tensor].size())) == 2:
              for layer in state_dict[tensor]:
                  dim_gen += len(layer)
          elif len(list(state_dict[tensor].size())) == 1:
              dim_gen += len(layer)
          else:
              print("!!!WARNING!!! Error in state dict dimensions")
    return dim_gen


def actor_to_genotype(actor):
    """ 
    Get the genotype corresponding to a DNN actor. 
    Inputs: actor {Actor} - the actor 
    Outputs: gen {list} - the list of genotype values
    """
    state_dict = actor.state_dict()
    gen = []
    for tensor in state_dict:
      if "weight" or "bias" in tensor:
          if len(list(state_dict[tensor].size())) == 2:
              for layer in state_dict[tensor]:
                  gen += layer.tolist()
          elif len(list(state_dict[tensor].size())) == 1:
              gen += layer.tolist()
          else:
              print("!!!WARNING!!! Error in state dict dimensions")
    return gen


def genotype_to_actor(gen, actor):
    """ 
    Fill in the DNN actor from a given genotype. 
    Inputs: 
        - gen {list} - the list of genotype values
        - actor {Actor} - the actor to fill in with gen values
    Outputs: actor {Actor} - the updated actor
    """
    state_dict = actor.state_dict()
    idx = 0
    for tensor in state_dict:
      if "weight" or "bias" in tensor:
          if len(list(state_dict[tensor].size())) == 2:
              for i in range (len(state_dict[tensor])):
                  for j in range (len(state_dict[tensor][i])):
                      state_dict[tensor][i][j] = gen[idx]
                      idx += 1
          elif len(list(state_dict[tensor].size())) == 1:
              for i in range (len(state_dict[tensor])):
                 state_dict[tensor][i] = gen[idx]
                 idx += 1
          else:
              print("!!!WARNING!!! Error in state dict dimensions")

    actor.load_state_dict(state_dict)
    return actor
