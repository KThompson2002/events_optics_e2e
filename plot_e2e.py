import csv
import matplotlib.pyplot as plt

iters, edge, activity, loss, nevents, mean_r, mean_t = [], [], [], [], [], [], []

with open("e2e_log.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        iters.append(int(row["iter"]))
        edge.append(float(row["edge"]))
        activity.append(float(row["activity"]))
        loss.append(float(row["loss"]))
        nevents.append(int(row["num_events"]))
        mean_r.append(float(row["mean_radius"]))
        mean_t.append(float(row["mean_thickness"]))

plt.figure()
plt.plot(iters, edge)
plt.title("Edge Energy (Event Sharpness)")
plt.xlabel("Iteration"); plt.ylabel("Edge Energy")
plt.show()

plt.figure()
plt.plot(iters, nevents)
plt.title("Number of Events")
plt.xlabel("Iteration"); plt.ylabel("Event Count")
plt.show()

plt.figure()
plt.plot(iters, mean_r)
plt.title("Mean Surface Radius (Lens Shape Change)")
plt.xlabel("Iteration"); plt.ylabel("Radius")
plt.show()

plt.figure()
plt.plot(iters, mean_t)
plt.title("Mean Thickness / Spacing")
plt.xlabel("Iteration"); plt.ylabel("Thickness")
plt.show()
