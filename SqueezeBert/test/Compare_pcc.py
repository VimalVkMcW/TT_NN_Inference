import torch
import numpy as np
import logging
logger = logging.getLogger(__name__)

class PCC :
    def comp_pcc(golden, calculated, pcc=1):
        if golden.dtype != calculated.dtype:
            calculated = calculated.type(golden.dtype)

        if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
            logger.warning("Both tensors are 'nan'")
            return True, f"PCC: {1.0}"

        if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
            logger.error("One tensor is all nan, the other is not.")
            return False, f"PCC: {0.0}"

        # Test if either is completely zero
        if torch.any(golden.bool()) != torch.any(calculated.bool()):
            logger.error("One tensor is all zero")
            return False, f"PCC: {0.0}"

        # For now, mask all infs and nans so that we check the rest... TODO
        golden = golden.clone()
        golden[
            torch.logical_or(
                torch.isnan(golden),
                torch.logical_or(torch.isinf(golden), torch.isneginf(golden)),
            )
        ] = 0
        calculated = calculated.clone()
        calculated[
            torch.logical_or(
                torch.isnan(calculated),
                torch.logical_or(torch.isinf(calculated), torch.isneginf(calculated)),
            )
        ] = 0

        if torch.equal(golden, calculated):
            return True, f"PCC: {1.0}"

        if golden.dtype == torch.bfloat16:
            golden = golden.type(torch.float32)
            calculated = calculated.type(torch.float32)
        cal_pcc = np.min(
            np.ma.corrcoef(
                np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
                np.ma.masked_invalid(torch.squeeze(calculated).detach().numpy()).flatten(),
            )
        )

        if isinstance(cal_pcc, np.ma.core.MaskedConstant):
            return True, f"PCC: {1.0}"

        return cal_pcc >= pcc, f"PCC: {cal_pcc}"
    
    def modify_state_dict_with_prefix(model, wanted_prefix):
        state_dict = model.state_dict()
        if(wanted_prefix == ''):
            return state_dict
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(wanted_prefix):
                # Remove the prefix from the key
                new_key = k[len(wanted_prefix):]
                new_state_dict[new_key] = v

        return new_state_dict
    
    def flatten_tuple(tup):
        if isinstance(tup, tuple):
            return torch.cat([PCC.flatten_tuple(item) for item in tup if isinstance(item, torch.Tensor)], dim=-1)
        else:
            return tup

