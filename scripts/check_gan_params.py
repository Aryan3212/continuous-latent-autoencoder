import torch
from models.discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    periods = [2, 3, 5, 7, 11]
    scales = 3
    
    mpd = MultiPeriodDiscriminator(periods, channels=24)
    msd = MultiScaleDiscriminator(scales, channels=12)
    
    m_p = count_parameters(mpd)
    s_p = count_parameters(msd)
    
    print(f"Current MPD: {m_p:,}")
    print(f"Current MSD: {s_p:,}")
    print(f"Current Total: {m_p + s_p:,}")

if __name__ == "__main__":
    main()
