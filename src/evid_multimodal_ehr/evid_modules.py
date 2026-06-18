# PyTorch
import torch
from torch import nn


class DSFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        W: torch.Tensor,
        beta: torch.Tensor,
        alpha: torch.Tensor,
        gamma: torch.Tensor,
        n_class: int,
        prototype_dim: int,
    ):
        device = inputs.device
        batch_size, _input_dim = inputs.size()

        # beta2: [prototype_dim, n_class]
        beta2 = beta**2
        # beta2_: [prototype_dim,]
        beta2_ = beta2.sum(1)

        # u: [prototype_dim, n_class]
        u = beta2 / (beta2_.unsqueeze(1) * torch.ones(1, n_class, device=device))
        # alphap: [prototype_dim, 1]
        alphap = 0.99 / (1 + torch.exp(-alpha))

        # d: [prototype_dim, batch_size]
        d = torch.zeros(prototype_dim, batch_size, device=device)
        # s: [prototype_dim, batch_size]
        s = torch.zeros(prototype_dim, batch_size, device=device)
        # expo: [prototype_dim, batch_size]
        expo = torch.zeros(prototype_dim, batch_size, device=device)

        # mk: mbar
        # mk : [n_class + 1, batch_size]
        mk = torch.cat(
            (
                torch.zeros((n_class, batch_size), device=device),
                torch.ones((1, batch_size), device=device),
            ),
            0,
        )

        for k in range(prototype_dim):
            # temp: [input_dim, batch_size]
            temp = inputs.transpose(0, 1) - torch.mm(
                W[k].unsqueeze(1), torch.ones((1, batch_size), device=device)
            )

            d[k] = 0.5 * (temp * temp).sum(0)
            expo[k] = torch.exp(-gamma[k] ** 2 * d[k])
            s[k] = alphap[k] * expo[k]
            # m: [n_class + 1, batch_size]
            m = torch.cat(
                (
                    u[k].unsqueeze(1) * s[k],
                    torch.ones((1, batch_size), device=device) - s[k],
                ),
                0,
            )

            # eq. (74)
            # t2: [n_class, batch_size]
            t2 = mk[:n_class] * (
                m[:n_class] + torch.ones((n_class, 1), device=device) * m[n_class]
            )
            # t3: [n_class, batch_size]
            t3 = m[:n_class] * (torch.ones((n_class, 1), device=device) * mk[n_class])
            # eq. (75)
            # t4: [1, batch_size]
            t4 = (mk[n_class]) * (m[n_class]).unsqueeze(0)
            mk = torch.cat((t2 + t3, t4), 0)

        # K: [batch_size,]
        K = mk.sum(0)
        # mk_n: [batch_size, class_dim + 1]
        mk_n = (mk / (torch.ones((n_class + 1, 1), device=device) * K)).transpose(0, 1)

        ctx.save_for_backward(inputs, W, beta, alpha, gamma, mk, d)
        ctx.n_class = n_class
        ctx.prototype_dim = prototype_dim

        return mk_n

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx, grad_output: torch.Tensor
    ):
        # inputs (_type_): [batch_size, input_dim]
        # W (_type_): [prototype_dim, input_dim]
        # BETA (_type_): [prototype_dim, n_class]
        # alpha (_type_): [prototype_dim, 1]
        # gamma (_type_): [prototype_dim, 1]
        # mk: [n_class + 1, batch_size]
        # d: [prototype_dim, batch_size]
        inputs, W, BETA, alpha, gamma, mk, d = ctx.saved_tensors
        n_class = ctx.n_class
        prototype_dim = ctx.prototype_dim

        grad_input = grad_W = grad_BETA = grad_alpha = grad_gamma = None

        (batch_size, input_dim) = inputs.size()
        device = inputs.device
        mu = 0  # regularization parameter (default=0)
        # 1 if optimization of prototype centers, 0 otherwise (default=1)
        iw = 1

        # grad_output_: [batch_size, n_class]
        grad_output_ = grad_output[:, :n_class] * batch_size * n_class
        # K: [1, batch_size]
        K = mk.sum(0, keepdim=True)
        # K2: [1, batch_size]
        K2 = K**2
        # BETA2: [prototype_dim, class_dim]
        BETA2 = BETA * BETA
        # beta2: [prototype_dim, 1]
        beta2 = BETA2.sum(1, keepdim=True)
        # U: [prototype_dim, class_dim]
        U = BETA2 / (beta2 * torch.ones(1, n_class, device=device))
        # alphap: [prototype_dim, 1]
        alphap = 0.99 / (1 + torch.exp(-alpha))  # 200*1

        # I: [prototype_dim, prototype_dim]
        I = torch.eye(n_class, device=device)

        # s: [prototype_dim, batch_size]
        s = torch.zeros(prototype_dim, batch_size, device=device)
        # expo: [prototype_dim, batch_size]
        expo = torch.zeros(prototype_dim, batch_size, device=device)
        # mm: [n_class + 1, batch_size]
        mm = torch.cat(
            (
                torch.zeros(n_class, batch_size, device=device),
                torch.ones(1, batch_size, device=device),
            ),
            0,
        )

        # dEdm: [n_class + 1, batch_size]
        dEdm = torch.zeros(n_class + 1, batch_size, device=device)

        # dU: [prototype_dim, n_class]
        dU = torch.zeros(prototype_dim, n_class, device=device)
        # Ds: [prototype_dim, batch_size]
        Ds = torch.zeros(prototype_dim, batch_size, device=device)
        # DW: [prototype_dim, input_dim]
        DW = torch.zeros(prototype_dim, input_dim, device=device)

        for p in range(n_class):
            dEdm[p] = (
                grad_output_.transpose(0, 1)
                * (
                    I[:, p].unsqueeze(1) * K
                    - mk[:n_class]
                    - 1
                    / n_class
                    * (torch.ones(n_class, 1, device=device) * mk[n_class])
                )
            ).sum(0) / K2

        dEdm[n_class] = (
            (
                grad_output_.transpose(0, 1)
                * (
                    -mk[:n_class]
                    + 1
                    / n_class
                    * torch.ones(n_class, 1, device=device)
                    * (K - mk[n_class])
                )
            ).sum(0)
        ) / K2

        for k in range(prototype_dim):
            expo[k] = torch.exp(-gamma[k] ** 2 * d[k])
            s[k] = alphap[k] * expo[k]

            # m: [n_class + 1, batch_size]
            m = torch.cat(
                (
                    U[k].unsqueeze(1) * s[k],
                    torch.ones(1, batch_size, device=device) - s[k],
                ),
                0,
            )

            mm[n_class] = mk[n_class] / m[n_class]
            # L: [n_class, batch_size]
            L = torch.ones(n_class, 1, device=device) * mm[n_class, :]  # L:m_M+1
            mm[:n_class] = (mk[:n_class] - L * m[:n_class]) / (
                m[:n_class] + torch.ones(n_class, 1, device=device) * m[n_class]
            )  # m_j
            # R: [n_class, batch_size]
            R = mm[:n_class] + L  # function 97,
            # A: [n_class, batch_size]
            A = R * torch.ones(n_class, 1, device=device) * s[k]  # function 97, s
            # B: [n_class, batch_size]
            B = (
                U[k].unsqueeze(1) * torch.ones(1, batch_size, device=device) * R
                - mm[:n_class]
            )

            dU[k] = torch.mean((A * dEdm[:n_class]).view(n_class, -1).permute(1, 0), 0)
            Ds[k] = (dEdm[:n_class] * B).sum(0) - (dEdm[n_class] * mm[n_class])

            # tt1: [1, batch_size]
            tt1 = (
                Ds[k]
                * (gamma[k] ** 2 * torch.ones(1, batch_size, device=device))
                * s[k]
            )
            # tt2: [batch_size, input_dim]
            tt2 = (torch.ones(batch_size, 1, device=device) * W[k]) - inputs  # - input
            # tt1: [1, batch_size]
            tt1 = tt1.view(1, -1)
            # tt2: [batch_size, input_dim]
            tt2 = tt2.transpose(0, 1).reshape(input_dim, batch_size).permute(1, 0)
            DW[k] = -torch.mm(tt1, tt2)

        # DW: [prototype_dim, input_dim]
        DW = iw * DW / batch_size
        # T: [prototype_dim, n_class]
        T = beta2 * torch.ones(1, n_class, device=device)
        # Dbeta: [prototype_dim, n_class]
        Dbeta = (2 * BETA / T**2) * (
            dU * (T - BETA2)
            - (dU * BETA2).sum(1).unsqueeze(1) * torch.ones(1, n_class, device=device)
            + dU * BETA2
        )
        # Dgamma: [prototype_dim, n_class]
        Dgamma = (
            -2
            * torch.mean(((Ds * d * s).view(prototype_dim, -1)).t(), 0).unsqueeze(1)
            * gamma
        )
        # Dalpha: [prototype_dim, 1]
        Dalpha = (
            torch.mean(((Ds * expo).view(prototype_dim, -1)).t(), 0).unsqueeze(1) + mu
        ) * (0.99 * (1 - alphap) * alphap)

        # Dinput: [batch_size, input_dim]
        Dinput = torch.zeros(batch_size, input_dim, device=device)
        # temp2: [prototype_dim, input_dim]
        temp2 = torch.zeros(prototype_dim, input_dim, device=device)

        for n in range(batch_size):
            for k in range(prototype_dim):
                # test7: [1, input_dim]
                test7 = inputs[n] - W[k].unsqueeze(0)
                # test9: [1, 1]
                test9 = (Ds[k, n] * (gamma[k] ** 2) * s[k, n]).unsqueeze(0).unsqueeze(1)

                temp2[k] = -prototype_dim * test9 * test7
                Dinput[n] = temp2.mean(0)

        if ctx.needs_input_grad[0]:
            grad_input = Dinput
        if ctx.needs_input_grad[1]:
            grad_W = DW
        if ctx.needs_input_grad[2]:
            grad_BETA = Dbeta
        if ctx.needs_input_grad[3]:
            grad_alpha = Dalpha
        if ctx.needs_input_grad[4]:
            grad_gamma = Dgamma

        return grad_input, grad_W, grad_BETA, grad_alpha, grad_gamma, None, None


class EvidNets(nn.Module):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()

        self.hparams = hparams

        if "d_hidden" not in self.hparams:
            d_input = kwargs["d_input"]
        else:
            d_input = self.hparams.d_hidden

        self.BETA = nn.Parameter(torch.Tensor(hparams.prototype_dim, hparams.n_class))
        self.alpha = nn.Parameter(torch.Tensor(hparams.prototype_dim, 1))
        self.gamma = nn.Parameter(torch.Tensor(hparams.prototype_dim, 1))
        self.W = nn.Parameter(torch.Tensor(hparams.prototype_dim, d_input))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.W)
        nn.init.normal_(self.BETA)
        nn.init.constant_(self.gamma, 0.1)
        nn.init.constant_(self.alpha, 0)

    def forward(self, inputs):
        """_summary_

        Args:
            input (_type_): _description_

        Returns:
            _type_: _description_
        """
        return DSFunction.apply(
            inputs,
            self.W,
            self.BETA,
            self.alpha,
            self.gamma,
            self.hparams.n_class,
            self.hparams.prototype_dim,
        )
