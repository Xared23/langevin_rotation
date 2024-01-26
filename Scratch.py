###########
# dist1 = np.ones(100)*1000
# dist2 =  np.ones(100)*1000
# dist1[0:5] = [-i**2 for i in np.linspace(-10,10,num=5)]
# dist2[90:] = [-i**2  for i in np.linspace(-10,10,num=10)]

# plt.plot(dist1)
# plt.plot(dist2)
# plt.show()

# p = [(np.max([0.00000000001, np.exp(-dist1[i])])) for i in range(len(dist1))]
# p = p / np.sum(p)
# q = [(np.max([0.00000000001, np.exp(-dist2[i])])) for i in range(len(dist2))]
# q = q / np.sum(q)
# #
# p = [(np.exp(-dist1[i])) for i in range(len(dist1))]
# p = p / np.sum(p)
# q = [(np.exp(-dist2[i])) for i in range(len(dist2))]
# q = q / np.sum(q)
a
# p = np.zeros(10000)
# q = np.zeros(10000)
# p[0:1000] = [-(i+1)*(i-1) for i in np.linspace(-1,1,num=1000)]
# p = p / np.sum(p)
# q[0:100] = [-(i+1)*(i-1) for i in np.linspace(-1,1,num=100)]
# q = q / np.sum(q)
# p_quants, q_quants = quantile_align(p,q)
#
# print(fit_djs(p_quants,q_quants))

# p_cumul = [np.trapz(p[0:i]) for i in range(len(p))]
# q_cumul = [np.trapz(q[0:i]) for i in range(len(q))]
# plt.plot(p_quants)
# plt.plot(q_quants)
# plt.show()
#

# plt.plot(p_quants,q_quants)
# plt.plot(p_quants,fit_comparison(p_quants, q_quants))
# plt.show()

# time_epoch = [(0,5) for i in range(50)]
# # spike_times = np.array([[np.array([rand.uniform(0,5) for k in range(20)]) for j in range(50)] for i in range(100)])
# spike_times = np.empty((100,50),dtype= np.ndarray)
# for i in range(100):
#     for j in range(50):
#         spike_times[i,j] = np.array([rand.uniform(0,5) for k in range(20)])
#
#
# data_ISI=em.transform_spikes_to_isi(spike_times, time_epoch)
