import os
import torch
from torch.utils.cpp_extension import load

pair_wise_distance_cuda = load(
    name='pair_wise_distance_cuda',
    sources=[os.path.join(os.path.dirname(__file__),
                          'pair_wise_distance_cuda_source.cu')],
    extra_cuda_cflags=['-O2']
)

class PairwiseDistFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, pixel_features, spixel_features, abs_spix_indinces, num_spixels_width, num_spixels_height):
        self.num_spixels_width = num_spixels_width
        self.num_spixels_height = num_spixels_height
        output = pixel_features.new(pixel_features.shape[0], 9, pixel_features.shape[-1]).zero_()
        self.save_for_backward(pixel_features, spixel_features, abs_spix_indinces) 

        return pair_wise_distance_cuda.forward(
            pixel_features.contiguous(), spixel_features.contiguous(),
            abs_spix_indinces.contiguous(), output,
            self.num_spixels_width, self.num_spixels_height)

    @staticmethod
    def backward(self, dist_matrix_grad):
        pixel_features, spixel_features, abs_spix_indinces = self.saved_tensors

        pixel_features_grad = torch.zeros_like(pixel_features)
        spixel_features_grad = torch.zeros_like(spixel_features)

        pixel_features_grad, spixel_features_grad = pair_wise_distance_cuda.backward(
            dist_matrix_grad.contiguous(), pixel_features.contiguous(),
            spixel_features.contiguous(),
            abs_spix_indinces.contiguous(),
            pixel_features_grad, spixel_features_grad,
            self.num_spixels_width, self.num_spixels_height
        )
        return pixel_features_grad, spixel_features_grad, None, None, None
