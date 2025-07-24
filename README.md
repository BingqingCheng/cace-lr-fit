# cace-lr-fit

## Code
The CACE code can be found in [https://github.com/BingqingCheng/cace].

Note: 
In the current work, we have optimized the LES part of the code in the CACE repository: we now first add up the short-range and the long-range energies using a FeatureAdd module and then apply the autograd of the total energy with respect to atomic positions to obtain forces. 
For comparison, the previous implementation uses two autograd operations to obtain short-range and long-range forces separately and then sums up the forces.
The elimination of one autograd operation significantly reduces computational cost.

An example of the new training script can be found in 'fit-water-timing'
Training data for pt-KF-aq is from [[https://doi.org/10.1103/48ct-3jxm](https://journals.aps.org/prl/abstract/10.1103/48ct-3jxm)]

Note: University of California, Berkeley has filed a provisional patent for the Latent Ewald Summation algorithm. No restriction on academic use.


## License

This project is licensed under the CC BY-NC 4.0 License.


