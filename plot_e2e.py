import csv
import matplotlib.pyplot as plt

iters, edge, activity, loss, num_events, mean_r, mean_t = [], [], [], [], [], [], []

with open("e2e_log.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        iters.append(int(row["iter"]))
        edge.append(float(row["edge"]))
        activity.append(float(row["activity"]))
        loss.append(float(row["loss"]))
        num_events.append(int(row["num_events"]))
        mean_r.append((row["mean_radius"]))
        mean_t.append((row["mean_thickness"]))


plt.figure()
plt.plot(iters, edge)
plt.xlabel("Iteration")
plt.ylabel("Edge energy")
plt.title("SENPI event-frame edge energy")
plt.show()

plt.figure()
plt.plot(iters, num_events)
plt.xlabel("Iteration")
plt.ylabel("Number of events")
plt.title("Event count (non-diff, for monitoring)")
plt.show()

plt.figure()
plt.plot(iters, activity)
plt.xlabel("Iteration")
plt.ylabel("Mean |event frame|")
plt.title("Event-frame activity")
plt.show()

plt.figure()
plt.plot(iters, mean_r)
plt.title("Mean Surface Radius (Lens Curvature Change)")
plt.xlabel("Iteration")
plt.ylabel("Radius")
plt.show()

plt.figure()
plt.plot(iters, mean_t)
plt.title("Mean Surface Thickness / Spacing")
plt.xlabel("Iteration")
plt.ylabel("Thickness")
plt.show()
