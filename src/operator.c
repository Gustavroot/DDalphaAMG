#include "operator.h"

unsigned int clover_site_size(unsigned int num_lattice_site_var, unsigned int depth) {
  if (depth == 0) {
    return 42;
  }
  return (num_lattice_site_var * (num_lattice_site_var + 1)) / 2;
}

unsigned int projection_buffer_size(unsigned int num_lattice_site_var,
                                    unsigned int num_lattice_sites) {
  return (num_lattice_site_var/2)*num_lattice_sites;
}
