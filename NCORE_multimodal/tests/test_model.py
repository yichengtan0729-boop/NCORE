import torch
from ncore.models.model import NCORE


def cfg():
    return {
      "paths": {"report_model_name_or_path":""},
      "experiment":{"task_names":["vent24","mortality"]},
      "model":{"num_concepts":6,"concept_dim":16,"hidden_dim":32,"max_steps":3,"operator_rho":.98,"operator_eta":.05,"modalities":["image","report","ehr"],"encoders":{
        "image":{"backend":"precomputed","input_dim":8,"output_dim":8},
        "report":{"backend":"precomputed","input_dim":10,"output_dim":10},
        "ehr":{"backend":"precomputed","input_dim":12,"output_dim":12}}},
      "data":{"report_max_length":64},
      "rl":{"reward":{"pred":1,"order":.2,"evidence":.1,"faithfulness":.1,"length":.05}}
    }


def batch():
    return {"modalities":{"image":torch.randn(4,8),"report":torch.randn(4,10),"ehr":torch.randn(4,12)},"modality_mask":torch.tensor([[1,1,1],[1,1,0],[1,0,1],[1,1,1.]],dtype=torch.float),"labels":torch.randint(0,2,(4,2)).float(),"label_mask":torch.ones(4,2),"meta":[{}]*4}


def test_rollout_shapes():
    m=NCORE(cfg()); b=batch(); ro=m.sample_rollout(b)
    assert ro["logits"].shape==(4,2); assert ro["actions"].shape==(4,3)
    assert torch.isfinite(ro["logits"]).all()


def test_noncommutative_order_can_differ():
    m=NCORE(cfg()); b=batch(); enc=m.encode(b); ops=m.build_operators(enc); s=m.initial(4,torch.device("cpu"))
    a=torch.tensor([0,0,0,0]); c=torch.tensor([m.K,m.K,m.K,m.K])
    s1,_=m.apply_action(s,enc,ops,a); s12,_=m.apply_action(s1,enc,ops,c)
    s2,_=m.apply_action(s,enc,ops,c); s21,_=m.apply_action(s2,enc,ops,a)
    assert (s12-s21).abs().mean() > 0
