import unittest

import torch

from gr_sat.models import TelemetryVAE, compute_anomaly_scores


class TelemetryVAETests(unittest.TestCase):
    def test_eval_mode_is_deterministic(self):
        torch.manual_seed(0)
        model = TelemetryVAE(input_dim=2, hidden_dim=4, latent_dim=2)
        model.eval()
        sample = torch.ones(3, 2)

        recon_a, mu_a, logvar_a = model(sample)
        recon_b, mu_b, logvar_b = model(sample)

        self.assertTrue(torch.allclose(recon_a, recon_b))
        self.assertTrue(torch.allclose(mu_a, mu_b))
        self.assertTrue(torch.allclose(logvar_a, logvar_b))

    def test_training_mode_samples_latent_noise(self):
        torch.manual_seed(0)
        model = TelemetryVAE(input_dim=2, hidden_dim=4, latent_dim=2)
        model.train()
        mu = torch.zeros(2, 2)
        logvar = torch.zeros(2, 2)

        latent = model.reparameterize(mu, logvar)

        self.assertFalse(torch.allclose(latent, mu))

    def test_compute_anomaly_scores_returns_per_sample_scores(self):
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        recon = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        mu = torch.zeros(2, 2)
        logvar = torch.zeros(2, 2)

        scores = compute_anomaly_scores(recon, inputs, mu, logvar, kld_weight=0.05)

        self.assertEqual(tuple(scores.shape), (2,))
        self.assertTrue(torch.allclose(scores, torch.ones(2)))


if __name__ == "__main__":
    unittest.main()
