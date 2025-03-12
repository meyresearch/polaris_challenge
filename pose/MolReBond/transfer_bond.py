from rdkit import Chem
from rdkit.Chem import rdFMCS

def remove_aromatic_tags(mol):
    """Remove aromatic flags from all atoms and bonds in a molecule."""
    rw_mol = Chem.RWMol(mol)
    for atom in rw_mol.GetAtoms():
        atom.SetIsAromatic(False)
    for bond in rw_mol.GetBonds():
        bond.SetIsAromatic(False)
        # bond.SetStereo(Chem.BondStereo.STEREONONE)  # Reset bond stereo
    return rw_mol.GetMol()

def transfer_bond_orders(mol, reference):
    """Transfer bond orders from a reference molecule to a target molecule.

    Args:
        mol (_type_): rdkit mol object from prediction
        reference (_type_): smiles of the reference

  

    Returns:
        rdkit mol : molecule with bond orders transferred from reference
    """
    assert type(mol) == Chem.Mol, "mol must be an RDKit molecule object."
    assert type(reference) == str, "reference must be a SMILES string."
   
   
   
    try:
        m2_non_aromatic = Chem.MolFromSmiles(reference)
    except Exception as e:
        raise ValueError("Failed to convert reference smiles to mol .") from e
    
    m2_non_aromatic = Chem.RemoveHs(m2_non_aromatic)
    Chem.Kekulize(m2_non_aromatic, clearAromaticFlags=True)
    mol = Chem.RemoveHs(mol)
    # Remove aromatic tags from mol1
    mol = remove_aromatic_tags(mol)
    
    # Find MCS for atom mapping
    mcs = rdFMCS.FindMCS(
        [mol, m2_non_aromatic],
        bondCompare=rdFMCS.BondCompare.CompareAny,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        matchValences=False,
        timeout=60
    )
    
    if mcs.canceled or mcs.numAtoms != mol.GetNumAtoms():
        raise ValueError("MCS failed to map all atoms. Check connectivity or symmetry.")
    
    mcs_pattern = Chem.MolFromSmarts(mcs.smartsString)
    matches_m1 = mol.GetSubstructMatches(mcs_pattern)
    matches_m2 = m2_non_aromatic.GetSubstructMatches(mcs_pattern)
    
    if not matches_m1 or not matches_m2:
        raise ValueError("No substructure matches found.")

    match_m1 = matches_m1[0]
    match_m2 = matches_m2[0]
    m2_to_m1 = {m2_idx: m1_idx for m1_idx, m2_idx in zip(match_m1, match_m2)}
    
    # Prepare editable molecule from mol1
    m1_edited = Chem.RWMol(mol)
    
    # ==== NEW: Transfer formal charges first ====
    for atom in m2_non_aromatic.GetAtoms():
        m1_idx = m2_to_m1.get(atom.GetIdx())
        if m1_idx is not None:
            m1_edited.GetAtomWithIdx(m1_idx).SetFormalCharge(atom.GetFormalCharge())
    
    # Remove existing bonds
    bonds_to_remove = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) 
                      for bond in m1_edited.GetBonds()]
    for begin, end in reversed(bonds_to_remove):
        m1_edited.RemoveBond(begin, end)
    
    # Add new bonds from mol2
    for bond in m2_non_aromatic.GetBonds():
        a2_start = bond.GetBeginAtomIdx()
        a2_end = bond.GetEndAtomIdx()
        a1_start = m2_to_m1.get(a2_start)
        a1_end = m2_to_m1.get(a2_end)
        current_num_bonds = m1_edited.GetNumBonds()
        
        if a1_start is not None and a1_end is not None:
            m1_edited.AddBond(a1_start, a1_end, bond.GetBondType())

    
    return m1_edited.GetMol()
