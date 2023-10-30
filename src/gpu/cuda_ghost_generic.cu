#include "alloc_control.h"
#include "cuda_complex.h"
#include "cuda_ghost_PRECISION.h"
#include "global_struct.h"
#include "util_macros.h"

extern "C" {
#include "cuda_miscellaneous.h"
}


__global__ void _boundary2buffer(cu_cmplx_PRECISION const *phi, cu_cmplx_PRECISION *buffer,
                                 int const *table, size_t offset, size_t num_boundary_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_boundary_sites * offset) {
    // there is no more data for this index
    return;
  }
  size_t idx_boundary_site = idx / offset;
  size_t idx_data = idx % offset;
  phi += table[idx_boundary_site] * offset;
  buffer[idx] = phi[idx_data];
}

__global__ void _buffer2boundary(cu_cmplx_PRECISION *phi, cu_cmplx_PRECISION const *buffer,
                                 int const *table, size_t offset, size_t num_boundary_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_boundary_sites * offset) {
    // there is no more data for this index
    return;
  }
  size_t idx_boundary_site = idx / offset;
  size_t idx_data = idx % offset;
  phi += table[idx_boundary_site] * offset;
  phi[idx_data] = buffer[idx];
}

void cuda_ghost_alloc_PRECISION(int buffer_size, cuda_comm_PRECISION_struct *c, level_struct *l) {
  int mu, nu, factor = 1;

  if (l->depth > 0) {
    c->offset = l->num_lattice_site_var;
  } else {
    c->offset = l->num_lattice_site_var / 2;
    if (g.method < 5) factor = 2;
  }

  if (buffer_size <= 0) {
    c->comm_start[0] = c->offset * l->num_inner_lattice_sites;
    c->comm_start[1] = c->offset * l->num_inner_lattice_sites;
    for (mu = 0; mu < 4; mu++) {
      if (mu > 0) {
        c->comm_start[2 * mu] = c->comm_start[2 * (mu - 1)] + buffer_size;
        c->comm_start[2 * mu + 1] = c->comm_start[2 * (mu - 1) + 1] + buffer_size;
      }
      buffer_size = c->offset;
      for (nu = 0; nu < 4; nu++) {
        if (nu != mu) {
          buffer_size *= l->local_lattice[nu];
        }
      }
      c->max_length[mu] = factor * buffer_size;
      CUDA_MALLOC(c->buffer_gpu[2 * mu], cu_cmplx_PRECISION, factor * buffer_size);
      CUDA_MALLOC(c->buffer_gpu[2 * mu + 1], cu_cmplx_PRECISION, factor * buffer_size);
#ifdef GPU2GPU_COMMS_VIA_CPUS
      MALLOC(c->buffer[2 * mu], complex_PRECISION, factor * buffer_size);
      MALLOC(c->buffer[2 * mu + 1], complex_PRECISION, factor * buffer_size);
      MALLOC(c->buffer2[2 * mu], complex_PRECISION, factor * buffer_size);
      MALLOC(c->buffer2[2 * mu + 1], complex_PRECISION, factor * buffer_size);
#endif
      c->in_use[2 * mu] = 0;
      c->in_use[2 * mu + 1] = 0;
    }
  } else {
    for (mu = 0; mu < 4; mu++) {
      c->max_length[mu] = buffer_size;
      CUDA_MALLOC(c->buffer_gpu[2 * mu], cu_cmplx_PRECISION, buffer_size);
      CUDA_MALLOC(c->buffer_gpu[2 * mu + 1], cu_cmplx_PRECISION, buffer_size);
#ifdef GPU2GPU_COMMS_VIA_CPUS
      CUDA_MALLOC(c->buffer[2 * mu], complex_PRECISION, buffer_size);
      CUDA_MALLOC(c->buffer[2 * mu + 1], complex_PRECISION, buffer_size);
      CUDA_MALLOC(c->buffer2[2 * mu], complex_PRECISION, buffer_size);
      CUDA_MALLOC(c->buffer2[2 * mu + 1], complex_PRECISION, buffer_size);
#endif
    }
  }

  // no vbuf here
}

void cuda_ghost_free_PRECISION(cuda_comm_PRECISION_struct *c, level_struct *l) {
  int mu;

  for (mu = 0; mu < 4; mu++) {
    CUDA_FREE(c->buffer_gpu[2 * mu], complex_PRECISION, c->max_length[mu]);
    CUDA_FREE(c->buffer_gpu[2 * mu + 1], complex_PRECISION, c->max_length[mu]);
#ifdef GPU2GPU_COMMS_VIA_CPUS
    CUDA_FREE(c->buffer[2 * mu], complex_PRECISION, c->max_length[mu]);
    CUDA_FREE(c->buffer[2 * mu + 1], complex_PRECISION, c->max_length[mu]);
    CUDA_FREE(c->buffer2[2 * mu], complex_PRECISION, c->max_length[mu]);
    CUDA_FREE(c->buffer2[2 * mu + 1], complex_PRECISION, c->max_length[mu]);
#endif
  }

  // no vbuf here
}

void cuda_ghost_sendrecv_init_PRECISION(const int type, cuda_comm_PRECISION_struct *c,
                                        level_struct *l) {
  int mu;

  if (type == _COARSE_GLOBAL) {
    c->comm = 1;
    for (mu = 0; mu < 4; mu++) {
      ASSERT(c->in_use[2 * mu] == 0);
      ASSERT(c->in_use[2 * mu + 1] == 0);
    }
  }
}

void cuda_ghost_sendrecv_PRECISION(cuda_vector_PRECISION phi, const int mu, const int dir,
                                   cuda_comm_PRECISION_struct *c, const int amount,
                                   level_struct *l) {
  // does not allow sending in both directions at the same time
  if (l->global_splitting[mu] > 1) {
    int *table = NULL, mu_dir = 2 * mu - MIN(dir, 0), offset = c->offset, length[2] = {0, 0},
        comm_start = 0, table_start = 0;
#ifdef GPU2GPU_COMMS_VIA_CPUS
    cuda_vector_PRECISION phi_pt;
#else
    cuda_vector_PRECISION buffer, phi_pt;
#endif

    if (amount == _FULL_SYSTEM) {
      length[0] = (c->num_boundary_sites[2 * mu]) * offset;
      length[1] = (c->num_boundary_sites[2 * mu + 1]) * offset;
      comm_start = c->comm_start[mu_dir];
      table_start = 0;
    } else if (amount == _EVEN_SITES) {
      length[0] = c->num_even_boundary_sites[2 * mu] * offset;
      length[1] = c->num_even_boundary_sites[2 * mu + 1] * offset;
      comm_start = c->comm_start[mu_dir];
      table_start = 0;
    } else if (amount == _ODD_SITES) {
      length[0] = c->num_odd_boundary_sites[2 * mu] * offset;
      length[1] = c->num_odd_boundary_sites[2 * mu + 1] * offset;
      comm_start = c->comm_start[mu_dir] + c->num_even_boundary_sites[mu_dir] * offset;
      table_start = c->num_even_boundary_sites[mu_dir];
    }

    ASSERT(c->in_use[mu_dir] == 0);
    c->in_use[mu_dir] = 1;

    if (MAX(length[0], length[1]) > c->max_length[mu]) {
      printf("CAUTION: my_rank: %d, not enough comm buffer\n", g.my_rank);
      fflush(0);
      cuda_ghost_free_PRECISION(c, l);
      cuda_ghost_alloc_PRECISION(MAX(length[0], length[1]), c, l);
    }

#ifdef GPU2GPU_COMMS_VIA_CPUS
    //buffer = c->buffer_gpu[mu_dir];
#else
    buffer = c->buffer_gpu[mu_dir];
#endif

    // dir = senddir
    if (dir == 1) {
      // data to be communicated is stored serially in the vector phi
      // recv target is a buffer
      // afterwards (in ghost_wait) the data has to be distributed onto the correct sites
      // touching the respective boundary in -mu direction

      phi_pt = phi + comm_start;
      if (length[1] > 0) {
        PROF_PRECISION_START(_OP_COMM);
#ifdef GPU2GPU_COMMS_VIA_CPUS
        MPI_Irecv(c->buffer[mu_dir], length[1], MPI_COMPLEX_PRECISION, l->neighbor_rank[2 * mu + 1], 2 * mu,
                  g.comm_cart, &(c->rreqs[2 * mu]));
#else
        MPI_Irecv(buffer, length[1], MPI_COMPLEX_PRECISION, l->neighbor_rank[2 * mu + 1], 2 * mu,
                  g.comm_cart, &(c->rreqs[2 * mu]));
#endif
        PROF_PRECISION_STOP(_OP_COMM, 1);
      }
      if (length[0] > 0) {
        PROF_PRECISION_START(_OP_COMM);
#ifdef GPU2GPU_COMMS_VIA_CPUS
        cuda_safe_call( cudaMemcpy( (vector_PRECISION)(c->buffer2[mu_dir]), (cuda_vector_PRECISION)(phi_pt),
                                    length[0]*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToHost ) );
        MPI_Isend(c->buffer2[mu_dir], length[0], MPI_COMPLEX_PRECISION, l->neighbor_rank[2 * mu], 2 * mu,
                  g.comm_cart, &(c->sreqs[2 * mu]));
#else
        MPI_Isend(phi_pt, length[0], MPI_COMPLEX_PRECISION, l->neighbor_rank[2 * mu], 2 * mu,
                  g.comm_cart, &(c->sreqs[2 * mu]));
#endif
        PROF_PRECISION_STOP(_OP_COMM, 0);
      }

    } else if (dir == -1) {
      // data to be communicated is stored on the sites touching the boundary in -mu direction
      // this data is gathered in a buffer in the correct ordering
      // which is required on the boundary of the vector phi
      int num_boundary_sites = length[1] / offset;

      table = c->boundary_table_gpu[2 * mu + 1] + table_start;
      constexpr size_t blockSize = 128;
      const size_t gridSize = minGridSizeForN(num_boundary_sites * offset, blockSize);
#ifdef GPU2GPU_COMMS_VIA_CPUS
      _boundary2buffer<<<gridSize, blockSize>>>(phi, c->buffer_gpu[mu_dir], table, offset, num_boundary_sites);
#else
      _boundary2buffer<<<gridSize, blockSize>>>(phi, buffer, table, offset, num_boundary_sites);
#endif
      cuda_safe_call(cudaDeviceSynchronize());

      phi_pt = phi + comm_start;

      if (length[0] > 0) {
        PROF_PRECISION_START(_OP_COMM);
#ifdef GPU2GPU_COMMS_VIA_CPUS
        MPI_Irecv(c->buffer2[mu_dir], length[0], MPI_COMPLEX_PRECISION, l->neighbor_rank[2 * mu], 2 * mu + 1,
                  g.comm_cart, &(c->rreqs[2 * mu + 1]));
#else
        MPI_Irecv(phi_pt, length[0], MPI_COMPLEX_PRECISION, l->neighbor_rank[2 * mu], 2 * mu + 1,
                  g.comm_cart, &(c->rreqs[2 * mu + 1]));
#endif
        PROF_PRECISION_STOP(_OP_COMM, 1);
      }
      if (length[1] > 0) {
        PROF_PRECISION_START(_OP_COMM);
#ifdef GPU2GPU_COMMS_VIA_CPUS
        cuda_safe_call( cudaMemcpy( (vector_PRECISION)(c->buffer[mu_dir]), (cuda_vector_PRECISION)(c->buffer_gpu[mu_dir]),
                                    length[1]*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToHost ) );
        MPI_Isend(c->buffer[mu_dir], length[1], MPI_COMPLEX_PRECISION, l->neighbor_rank[2 * mu + 1],
                  2 * mu + 1, g.comm_cart, &(c->sreqs[2 * mu + 1]));
#else
        MPI_Isend(buffer, length[1], MPI_COMPLEX_PRECISION, l->neighbor_rank[2 * mu + 1],
                  2 * mu + 1, g.comm_cart, &(c->sreqs[2 * mu + 1]));
#endif
        PROF_PRECISION_STOP(_OP_COMM, 0);
      }

    } else
      ASSERT(dir == 1 || dir == -1);
  }
}

void cuda_ghost_wait_PRECISION(cuda_vector_PRECISION phi, const int mu, const int dir,
                               cuda_comm_PRECISION_struct *c, const int amount, level_struct *l) {
  if (l->global_splitting[mu] > 1) {
    int mu_dir = 2 * mu - MIN(dir, 0);
    int *table, offset = c->offset, length[2] = {0, 0}, table_start = 0;
#ifdef GPU2GPU_COMMS_VIA_CPUS
    //cuda_vector_PRECISION buffer;
#else
    cuda_vector_PRECISION buffer;
#endif

    if (amount == _FULL_SYSTEM) {
      length[0] = (c->num_boundary_sites[2 * mu]) * offset;
      length[1] = (c->num_boundary_sites[2 * mu + 1]) * offset;
      table_start = 0;
    } else if (amount == _EVEN_SITES) {
      length[0] = c->num_even_boundary_sites[2 * mu] * offset;
      length[1] = c->num_even_boundary_sites[2 * mu + 1] * offset;
      table_start = 0;
    } else if (amount == _ODD_SITES) {
      length[0] = c->num_odd_boundary_sites[2 * mu] * offset;
      length[1] = c->num_odd_boundary_sites[2 * mu + 1] * offset;
      table_start = c->num_even_boundary_sites[mu_dir];
    }

#ifdef GPU2GPU_COMMS_VIA_CPUS
    int comm_start = 0;
    cuda_vector_PRECISION phi_pt;

    if (amount == _FULL_SYSTEM) {
      comm_start = c->comm_start[mu_dir];
    } else if (amount == _EVEN_SITES) {
      comm_start = c->comm_start[mu_dir];
    } else if (amount == _ODD_SITES) {
      comm_start = c->comm_start[mu_dir] + c->num_even_boundary_sites[mu_dir] * offset;
    }

    phi_pt = phi + comm_start;
#endif

    ASSERT(c->in_use[mu_dir] == 1);

    if (dir == 1) {
      int num_boundary_sites = length[0] / offset;

#ifdef GPU2GPU_COMMS_VIA_CPUS
      //buffer = c->buffer_gpu[mu_dir];
#else
      buffer = c->buffer_gpu[mu_dir];
#endif
      table = c->boundary_table_gpu[2 * mu + 1] + table_start;

      if (length[0] > 0) {
        PROF_PRECISION_START(_OP_IDLE);
        MPI_Wait(&(c->sreqs[2 * mu]), MPI_STATUS_IGNORE);
        PROF_PRECISION_STOP(_OP_IDLE, 0);
      }
      if (length[1] > 0) {
        PROF_PRECISION_START(_OP_IDLE);
        MPI_Wait(&(c->rreqs[2 * mu]), MPI_STATUS_IGNORE);
#ifdef GPU2GPU_COMMS_VIA_CPUS
        cuda_safe_call( cudaMemcpy( (cuda_vector_PRECISION)(c->buffer_gpu[mu_dir]), (vector_PRECISION)(c->buffer[mu_dir]),
                                    length[1]*sizeof(cu_cmplx_PRECISION), cudaMemcpyHostToDevice ) );
#endif
        PROF_PRECISION_STOP(_OP_IDLE, 1);
      }
      constexpr size_t blockSize = 128;
      const size_t gridSize = minGridSizeForN(num_boundary_sites * offset, blockSize);
#ifdef GPU2GPU_COMMS_VIA_CPUS
      _buffer2boundary<<<gridSize, blockSize>>>(phi, c->buffer_gpu[mu_dir], table, offset, num_boundary_sites);
#else
      _buffer2boundary<<<gridSize, blockSize>>>(phi, buffer, table, offset, num_boundary_sites);
#endif
      cuda_safe_call(cudaDeviceSynchronize());
    } else if (dir == -1) {
      if (length[1] > 0) {
        PROF_PRECISION_START(_OP_IDLE);
        MPI_Wait(&(c->sreqs[2 * mu + 1]), MPI_STATUS_IGNORE);
        PROF_PRECISION_STOP(_OP_IDLE, 0);
      }
      if (length[0] > 0) {
        PROF_PRECISION_START(_OP_IDLE);
        MPI_Wait(&(c->rreqs[2 * mu + 1]), MPI_STATUS_IGNORE);
#ifdef GPU2GPU_COMMS_VIA_CPUS
        cuda_safe_call( cudaMemcpy( (cuda_vector_PRECISION)(phi_pt), (vector_PRECISION)(c->buffer2[mu_dir]),
                                    length[0]*sizeof(cu_cmplx_PRECISION), cudaMemcpyHostToDevice ) );
#endif
        PROF_PRECISION_STOP(_OP_IDLE, 1);
      }

    } else
      ASSERT(dir == 1 || dir == -1);

    c->in_use[mu_dir] = 0;
  }
}

extern "C" void cuda_ghost_update_PRECISION(cuda_vector_PRECISION phi, const int mu, const int dir,
                                            comm_PRECISION_struct *c, level_struct *l) {
  if (l->global_splitting[mu] > 1) {
    int mu_dir = 2 * mu - MIN(dir, 0), nu, inv_mu_dir = 2 * mu + 1 + MIN(dir, 0), length,
        comm_start, num_boundary_sites;
    cuda_vector_PRECISION buffer, recv_pt;

    length = c->num_boundary_sites[mu_dir] * l->num_lattice_site_var;
    num_boundary_sites = c->num_boundary_sites[mu_dir];
#ifdef GPU2GPU_COMMS_VIA_CPUS
    //buffer = (cuda_vector_PRECISION)c->buffer_gpu[mu_dir];
#else
    buffer = (cuda_vector_PRECISION)c->buffer_gpu[mu_dir];
#endif

    if (dir == -1)
      comm_start = l->vector_size;
    else
      comm_start = l->inner_vector_size;
    for (nu = 0; nu < mu; nu++) {
      comm_start += c->num_boundary_sites[2 * nu] * l->num_lattice_site_var;
    }

    ASSERT(c->in_use[mu_dir] == 0);
    c->in_use[mu_dir] = 1;

    recv_pt = phi + comm_start;
    if (length > 0) {
      PROF_PRECISION_START(_OP_COMM);
#ifdef GPU2GPU_COMMS_VIA_CPUS
      MPI_Irecv(c->buffer2[mu_dir], length, MPI_COMPLEX_PRECISION, l->neighbor_rank[mu_dir], mu_dir,
                g.comm_cart, &(c->rreqs[mu_dir]));
#else
      MPI_Irecv(recv_pt, length, MPI_COMPLEX_PRECISION, l->neighbor_rank[mu_dir], mu_dir,
                g.comm_cart, &(c->rreqs[mu_dir]));
#endif
      PROF_PRECISION_STOP(_OP_COMM, 1);
    }
    constexpr size_t blockSize = 128;
    constexpr size_t offset = 12;
    const size_t gridSize = minGridSizeForN(num_boundary_sites * offset, blockSize);
#ifdef GPU2GPU_COMMS_VIA_CPUS
    _boundary2buffer<<<gridSize, blockSize>>>(phi, c->buffer_gpu[mu_dir], c->boundary_table_gpu[inv_mu_dir],
                                              offset, num_boundary_sites);
#else
    _boundary2buffer<<<gridSize, blockSize>>>(phi, buffer, c->boundary_table_gpu[inv_mu_dir],
                                              offset, num_boundary_sites);
#endif
    cuda_safe_call(cudaDeviceSynchronize());

    if (length > 0) {
      PROF_PRECISION_START(_OP_COMM);
#ifdef GPU2GPU_COMMS_VIA_CPUS
      cuda_safe_call( cudaMemcpy( (vector_PRECISION)(c->buffer[mu_dir]), (cuda_vector_PRECISION)(c->buffer_gpu[mu_dir]),
                                  length*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToHost ) );
      MPI_Isend(c->buffer[mu_dir], length, MPI_COMPLEX_PRECISION, l->neighbor_rank[inv_mu_dir], mu_dir,
                g.comm_cart, &(c->sreqs[mu_dir]));
#else
      MPI_Isend(buffer, length, MPI_COMPLEX_PRECISION, l->neighbor_rank[inv_mu_dir], mu_dir,
                g.comm_cart, &(c->sreqs[mu_dir]));
#endif
      PROF_PRECISION_STOP(_OP_COMM, 0);
    }
  }
}

extern "C" void cuda_ghost_update_wait_PRECISION(cuda_vector_PRECISION phi, const int mu,
                                                 const int dir, comm_PRECISION_struct *c,
                                                 level_struct *l) {
  if (l->global_splitting[mu] > 1) {
    int mu_dir = 2 * mu - MIN(dir, 0),
#ifdef GPU2GPU_COMMS_VIA_CPUS
        length = c->num_boundary_sites[mu_dir] * l->num_lattice_site_var,nu;
#else
        length = c->num_boundary_sites[mu_dir] * l->num_lattice_site_var;
#endif

    ASSERT(c->in_use[mu_dir] == 1);

#ifdef GPU2GPU_COMMS_VIA_CPUS
    int comm_start;
    cuda_vector_PRECISION recv_pt;

    if (dir == -1)
      comm_start = l->vector_size;
    else
      comm_start = l->inner_vector_size;
    for (nu = 0; nu < mu; nu++) {
      comm_start += c->num_boundary_sites[2 * nu] * l->num_lattice_site_var;
    }

    recv_pt = phi + comm_start;
#endif

    if (length > 0) {
      PROF_PRECISION_START(_OP_IDLE);
      MPI_Wait(&(c->sreqs[mu_dir]), MPI_STATUS_IGNORE);
      MPI_Wait(&(c->rreqs[mu_dir]), MPI_STATUS_IGNORE);
#ifdef GPU2GPU_COMMS_VIA_CPUS
        cuda_safe_call( cudaMemcpy( (cuda_vector_PRECISION)(recv_pt), (vector_PRECISION)(c->buffer2[mu_dir]),
                                    length*sizeof(cu_cmplx_PRECISION), cudaMemcpyHostToDevice ) );
#endif
      PROF_PRECISION_STOP(_OP_IDLE, 1);
    }
    c->in_use[mu_dir] = 0;
  }
}
