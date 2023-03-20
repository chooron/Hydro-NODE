# scipy≤Â÷µø‚
# from scipy.interpolate import PchipInterpolator
#
# t_series_np = np.linspace(0, len(precp_series) - 1, len(precp_series))
# precp_spline = PchipInterpolator(t_series_np, precp_series)
# temp_spline = PchipInterpolator(t_series_np, temp_series)
# lday_spline = PchipInterpolator(t_series_np, lday_series)

# torchcde≤Â÷µø‚
# from torchcde import hermite_cubic_coefficients_with_backward_differences, CubicSpline
#
# t_series = torch.linspace(0, len(precp_series) - 1, len(precp_series))
# precp_spline = CubicSpline(hermite_cubic_coefficients_with_backward_differences(
#     torch.from_numpy(precp_series).unsqueeze(1), t_series))
# temp_spline = CubicSpline(hermite_cubic_coefficients_with_backward_differences(
#     torch.from_numpy(temp_series).unsqueeze(1), t_series))
# lday_spline = CubicSpline(hermite_cubic_coefficients_with_backward_differences(
#     torch.from_numpy(lday_series).unsqueeze(1), t_series))

# from torchcde import linear_interpolation_coeffs, LinearInterpolation
#
# t_series = torch.linspace(0, len(precp_series) - 1, len(precp_series))
# precp_spline = LinearInterpolation(linear_interpolation_coeffs(
#     torch.from_numpy(precp_series).unsqueeze(1), t_series))
# temp_spline = LinearInterpolation(linear_interpolation_coeffs(
#     torch.from_numpy(temp_series).unsqueeze(1), t_series))
# lday_spline = LinearInterpolation(linear_interpolation_coeffs(
#     torch.from_numpy(lday_series).unsqueeze(1), t_series))