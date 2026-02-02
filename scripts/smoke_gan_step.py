import torch

from models.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_matching_loss,
    generator_loss,
)


def main() -> None:
    mpd = MultiPeriodDiscriminator([2, 3, 5])
    msd = MultiScaleDiscriminator(scales=2)
    wav_real = torch.randn(2, 1, 512)
    wav_fake = torch.randn(2, 1, 512, requires_grad=True)

    real_mpd, fmap_real_mpd = mpd(wav_real)
    fake_mpd, fmap_fake_mpd = mpd(wav_fake.detach())
    real_msd, fmap_real_msd = msd(wav_real)
    fake_msd, fmap_fake_msd = msd(wav_fake.detach())
    d_loss = discriminator_loss(real_mpd, fake_mpd) + discriminator_loss(real_msd, fake_msd)

    fake_mpd_g, fmap_fake_mpd_g = mpd(wav_fake)
    fake_msd_g, fmap_fake_msd_g = msd(wav_fake)
    g_adv = generator_loss(fake_mpd_g) + generator_loss(fake_msd_g)
    fm = feature_matching_loss(fmap_real_mpd, fmap_fake_mpd_g) + feature_matching_loss(
        fmap_real_msd, fmap_fake_msd_g
    )
    g_loss = g_adv + fm
    g_loss.backward()

    assert wav_fake.grad is not None
    print("gan smoke step passed")


if __name__ == "__main__":
    main()
