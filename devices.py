from datetime import datetime
from config import (
    ACTION_CLEAR_TRAFFIC, ACTION_ON_VEHICLE, ACTUATOR_TYPE,
    BUSY_POWER1, BUSY_POWER2, BUSY_POWER3,
    CLOUD_COST_UNIT_TIME, CLOUD_CPU_SPEED,
    CPU_SPEED1, CPU_SPEED2, CPU_SPEED3,
    DATA_SIZE1_MIN, DATA_SIZE1_MAX,
    DATA_SIZE2_MIN, DATA_SIZE2_MAX,
    DATA_SIZE3_MIN, DATA_SIZE3_MAX,
    DATA_SIZE4_MIN, DATA_SIZE4_MAX,
    DEVICE_TO_CLASS,
    FD_BANDWIDTH1, FD_BANDWIDTH2, FD_BANDWIDTH3,
    FD_RAM1, FD_RAM2, FD_RAM3,
    IDLE_POWER1, IDLE_POWER2, IDLE_POWER3,
    LATENCY_CS_FR, LATENCY_ED_FR,
    MAX_SIMULATION_TIME,
    NO_INSTRUCTIONS1_MIN, NO_INSTRUCTIONS1_MAX,
    NO_INSTRUCTIONS2_MIN, NO_INSTRUCTIONS2_MAX,
    NO_INSTRUCTIONS3_MIN, NO_INSTRUCTIONS3_MAX,
    NO_INSTRUCTIONS4_MIN, NO_INSTRUCTIONS4_MAX,
    NUM_FOG_NODES,
    RAM_SWAP_UNIT, RANDOM_SEED, SENSOR_TYPE,
    SLOT_TIME, TRAFFIC_LOCATION, VEHICLE_IDENTIFICATION
)
import random
random.seed(RANDOM_SEED)

g_jobserno = 0
g_taskid = 0
g_devid = 0

jobQ = []
ClusterRs = []
Sensors = []
Actuators = []

resource_mg_string = ['DEFAULT_RM', 'GA_RM', 'AI_RM', 'RL_RM', 'RR_RM', 'DAGGER_RM', 'PSO_RM']

taskType_strings = [
    'CAMERA_SEN', 'TRAFFIC_SEN', 'VEHICLE_IDENTIFICATION',
    'TRAFFIC_LOCATION', 'ACTION_ON_VEHICLE', 'ACTION_CLEAR_TRAFFIC'
]


class DeviceTime:
    def __init__(self):
        self.tm = 0.

    def set_time(self, tm):
        self.tm = tm

    def add_time(self, tm):
        self.tm += tm

    def get_time(self):
        return self.tm


class DeviceJobQ:
    def __init__(self):
        self.tq = []

    def add_t(self, tsk):
        self.tq.append(tsk)

    def next_t(self):
        tsk = self.tq.pop()
        return tsk


class TimeTrackerJb:
    def __init__(self):
        self.maxTimetakenbyJob = 0.
        self.minTimetakenbyJob = 1000.
        self.sumTimetakenbyJob = 0.
        self.no_jobs = 0

    def RecordTimeDetails(self, jb):
        if jb.comp_tmstamp > self.maxTimetakenbyJob:
            self.maxTimetakenbyJob = jb.comp_tmstamp
        if jb.comp_tmstamp < self.minTimetakenbyJob:
            self.minTimetakenbyJob = jb.comp_tmstamp
        self.sumTimetakenbyJob = self.sumTimetakenbyJob + jb.comp_tmstamp
        self.no_jobs += 1

    def PrintTimeDetails(self, strng='Jobs:'):
        avgTimetakenbyJob = self.sumTimetakenbyJob / self.no_jobs
        print("Total Number of Jobs:" + str(self.no_jobs))
        print()
        print("max Timetaken for " + strng + str(self.maxTimetakenbyJob))
        print("min Timetaken for " + strng + str(self.minTimetakenbyJob))
        print("avg TimeTaken for " + strng + str(avgTimetakenbyJob))


class TimeTrackerTsk:
    def __init__(self):
        self.maxTimetakenbyTsk = 0.
        self.minTimetakenbyTsk = 1000.
        self.sumTimetakenbyTsk = 0.
        self.no_tasks = 0

    def RecordTimeDetails(self, tsk):
        if tsk.time_taken > self.maxTimetakenbyTsk:
            self.maxTimetakenbyTsk = tsk.time_taken
        if tsk.time_taken < self.minTimetakenbyTsk:
            self.minTimetakenbyTsk = tsk.time_taken
        self.sumTimetakenbyTsk = self.sumTimetakenbyTsk + tsk.time_taken
        self.no_tasks += 1

    def PrintTimeDetails(self, strng='Tasks:'):
        avgTimetakenbyTsks = self.sumTimetakenbyTsk / self.no_tasks
        print("Total Number of " + strng + str(self.no_tasks))
        print()
        print("max Timetaken for " + strng + str(self.maxTimetakenbyTsk))
        print("min Timetaken for " + strng + str(self.minTimetakenbyTsk))
        print("avg TimeTaken for " + strng + str(avgTimetakenbyTsks))


class Task:
    def __init__(self, jbid, tsktype, sr, slotno):
        global g_taskid
        g_taskid += 1
        self.id = g_taskid

        self.job_id = jbid
        self.task_type = tsktype
        self.slot_no = slotno

        self.fd_devid = 0
        self.sensor_id = sr

        self.duration = 0.
        self.delay = 0
        self.sb_timestamp = 0.
        self.time_taken = 0.

        self.destId = 0
        self.dataBytes = 0

        match tsktype:
            case 1:
                self.no_instructions = random.randint(
                    NO_INSTRUCTIONS1_MIN, NO_INSTRUCTIONS1_MAX)
                self.dataBytes = random.randint(
                    DATA_SIZE2_MIN, DATA_SIZE2_MAX)
            case 2:
                self.no_instructions = random.randint(
                    NO_INSTRUCTIONS2_MIN, NO_INSTRUCTIONS2_MAX)
                self.dataBytes = random.randint(
                    DATA_SIZE1_MIN, DATA_SIZE1_MAX)
            case 3:
                self.no_instructions = random.randint(
                    NO_INSTRUCTIONS3_MIN, NO_INSTRUCTIONS3_MAX)
                self.dataBytes = random.randint(
                    DATA_SIZE3_MIN, DATA_SIZE3_MAX)
            case 4:
                self.no_instructions = random.randint(
                    NO_INSTRUCTIONS4_MIN, NO_INSTRUCTIONS4_MAX)
                self.dataBytes = random.randint(
                    DATA_SIZE4_MIN, DATA_SIZE4_MAX)
            case 5:
                self.no_instructions = 0
                self.dataBytes = 0
            case _:
                self.no_instructions = 0
                self.dataBytes = 0

    def get_noInstructions(self):
        return self.no_instructions

    def get_dataBytes(self):
        return self.dataBytes

    def get_tasktype(self):
        return self.task_type - 1

    def get_devid(self):
        return self.sensor_id

    def writeDetails(self):
        global logentry
        self.time_taken = self.duration + self.delay
        strVal = ('slot Id:' + str(self.slot_no) +
                  ' TaskId:' + str(self.id) +
                  ' JobId:' + str(self.job_id) +
                  ' CreaterId:' + str(self.sensor_id) +
                  ' ClusterId:' + str(self.destId) +
                  ' submited at:' + str(self.sb_timestamp) +
                  ' time taken:' + str(self.time_taken) + '\n')
        logentry.writeTaskDetails(strVal)

    def setSlotNo(self, sn):
        self.slot_no = sn

    def setSubmitime(self, sim_time):
        self.sb_timestamp = sim_time

    def setExecutionstarttime(self, delay):
        self.ex_timestamp = self.sb_timestamp + delay
        self.delay = delay

    def setExecutionduration(self, d):
        self.duration = d

    def get_timetaken(self):
        return self.time_taken


class Job:
    def __init__(self, src, fd, sensertype, tm, slot_no):
        global g_jobserno
        g_jobserno += 1
        self.id = g_jobserno

        self.sorceSenser = src
        self.fd_devid = fd
        self.sensertype = sensertype

        self.slot_no = slot_no
        self.slot_noCr = slot_no
        self.slot_noFi = 0

        self.sb_timestamp = tm
        self.comp_tmstamp = 0
        self.NoValidTask = False

        self.tsk = Task(self.id, self.sensertype, src, slot_no)
        self.tsk.setSubmitime(tm)
        self.ex_dev_id = fd

    def getDestinatinFogID(self):
        return self.fd_devid

    def setSubmitime(self, sim_time):
        self.sb_timestamp = sim_time

    def setSlotNo(self, sn):
        self.slot_no = sn

    def set_Cr_SlotNo(self, sn):
        self.slot_noCr = sn

    def set_ex_devid(self, id):
        self.ex_dev_id = id

    def get_ex_devid(self):
        return self.ex_dev_id

    def get_ex_class_label(self):
        """Returns the class label for the executing device.
        Uses the explicit DEVICE_TO_CLASS mapping instead of
        relying on device IDs coincidentally matching class indices.
        """
        return DEVICE_TO_CLASS.get(self.ex_dev_id, 0)

    def set_Fi_SlotNo(self, sn):
        self.slot_noFi = sn

    def setExecutionDetails(self, delay, swap_time, dst, speed, slotno):
        self.tsk.setExecutionstarttime(delay)
        self.tsk.destId = dst
        duration = self.tsk.no_instructions / speed + swap_time
        self.tsk.setExecutionduration(duration)
        self.tsk.writeDetails()
        tasksTimetracker.RecordTimeDetails(self.tsk)

        self.comp_tmstamp = self.comp_tmstamp + self.tsk.time_taken

        new_tasktype = ACTUATOR_TYPE
        match self.tsk.task_type:
            case 1:
                new_tasktype = VEHICLE_IDENTIFICATION
                self.tsk = Task(
                    self.id, new_tasktype, self.tsk.destId, slotno + 1)
                self.tsk.setSubmitime(slotno * SLOT_TIME)
            case 2:
                new_tasktype = TRAFFIC_LOCATION
                self.tsk = Task(
                    self.id, new_tasktype, self.tsk.destId, slotno + 1)
                self.tsk.setSubmitime(slotno * SLOT_TIME)
            case 3:
                new_tasktype = ACTION_ON_VEHICLE
                self.tsk = Task(
                    self.id, new_tasktype, self.tsk.destId, slotno + 1)
                self.tsk.setSubmitime(slotno * SLOT_TIME)
            case 4:
                new_tasktype = ACTION_CLEAR_TRAFFIC
                self.tsk = Task(
                    self.id, new_tasktype, self.tsk.destId, slotno + 1)
                self.tsk.setSubmitime(slotno * SLOT_TIME)
            case 5:
                self.NoValidTask = True
            case 6:
                self.NoValidTask = True
        return duration

    def get_noInstructions(self):
        return self.tsk.get_noInstructions()

    def get_type(self):
        return self.tsk.get_tasktype()

    def get_dataBytes(self):
        return self.tsk.get_dataBytes()

    def get_size(self):
        return self.get_noInstructions()

    def get_codesize(self):
        return self.get_noInstructions()

    def get_devise(self):
        return self.tsk.get_devid()


class FileWrite:
    def __init__(self):
        current_time = datetime.now()
        dt_string = current_time.strftime("%d-%m-%Y%H-%M-%S")
        self.fnameSlots = "slots-" + dt_string + ".txt"
        self.fnameTasks = "tasks-" + dt_string + ".txt"
        self.dataset = "dataset-" + dt_string + ".txt"
        print(self.fnameTasks)

    def writeTaskDetails(self, str):
        with open(self.fnameTasks, 'a') as f:
            f.write(str)

    def writeToDataset(self, str):
        with open(self.dataset, 'a') as f:
            f.write(str)

    def WriteStringToJobsLog(self, jb):
        with open(self.fnameTasks, 'a') as f:
            strval = ("Id:" + str(jb.id) +
                      " SlotNo:" + str(jb.slotno) +
                      " Source:" + jb.ed_name +
                      " Size:" + str(jb.no_instructions) +
                      " Duration:" + str(jb.duration) +
                      " SubmitTime:" + str(jb.sb_timestamp) +
                      " Destination:" + jb.dest +
                      " Executed at:" + str(jb.ex_timestamp) +
                      " Delay:" + str(
                          jb.ex_timestamp + jb.duration - jb.sb_timestamp) +
                      "\n")
            f.write(strval)

    def WriteStringToSlotsLog(self, str):
        with open(self.fnameSlots, 'a') as f:
            f.write(str)


class Actuator:
    def __init__(self, n, type=0):
        global g_devid
        g_devid += 1
        self.id = g_devid
        self.tmO = DeviceTime()
        self.name = n
        self.type = type

    def setClusterId(self, n):
        self.connected_cl_id = n

    def ExecutesTask(self, tsk):
        tsk.writeDetails()


class Sensor:
    def __init__(self, n, d_type):
        self.name = n
        self.task_type = d_type

        global g_devid
        g_devid += 1
        self.id = g_devid

    def setClusterId(self, n):
        self.connected_cl_id = n

    def CreateUploadJob(self, tm, slot_no):
        job = Job(self.id, self.connected_cl_id, self.task_type, tm, slot_no)
        return job


def compute_swap_time(no_instructions, data_bytes, ram):
    """Compute RAM swap time for a given task on a given device.
    Centralised to ensure GA, simulator, and dataset collection
    all use the same formula. Returns time in SU."""
    swap_ram_no = (no_instructions + data_bytes) / ram
    if swap_ram_no > 1:
        return RAM_SWAP_UNIT * swap_ram_no
    return RAM_SWAP_UNIT


def compute_exec_duration(no_instructions, cpu_speed, swap_time):
    """Compute total execution duration in TIME units (SU).
    Centralised to ensure GA and simulator agree on units.
    Returns: instructions / speed + swap_time (both in SU)."""
    return no_instructions / cpu_speed + swap_time


class Cloud:
    def __init__(self):
        global g_devid
        g_devid += 1
        self.id = g_devid

        self.devTime = DeviceTime()
        self.name = "Cloud Server"

        self.slot_no_task_ex = 0
        self.total_job_count = 0
        self.busy_time = 0
        self.slot_busy_time = 0

        self.cost_per_unit_time = CLOUD_COST_UNIT_TIME
        self.cpu_speed = CLOUD_CPU_SPEED
        self.netwrkutil = 0.

    def get_CPUSpeed(self):
        return self.cpu_speed

    def ExecutesJob(self, jb, sim_time, slotno):
        if self.devTime.get_time() < sim_time + SLOT_TIME:
            self.slot_no_task_ex += 1
            jb.set_ex_devid(self.id)
            swap_time = RAM_SWAP_UNIT
            duration = jb.setExecutionDetails(
                LATENCY_CS_FR + self.devTime.get_time() - sim_time,
                swap_time, self.id, self.cpu_speed, slotno)
            self.devTime.add_time(duration)
            self.slot_busy_time = self.slot_busy_time + duration

    def getIntervelMetricsDetails(self):
        return LATENCY_CS_FR + 1

    def clear(self, sim_time):
        self.total_job_count = self.total_job_count + self.slot_no_task_ex
        self.slot_no_task_ex = 0
        self.busy_time += self.slot_busy_time
        self.slot_busy_time = 0
        self.devTime.set_time(sim_time)

    def WriteSlotSummary(self):
        logentry.WriteStringToSlotsLog(
            "Cloud Executed:" + str(self.slot_no_task_ex) + "Tasks \n")

    def PrintNoTasksExecuted(self):
        print("Cloud executed :" + str(self.total_job_count) +
              " Tasks" +
              "\n       Time Used:" + str(self.busy_time) +
              "\n       Cost:" + str(
                  self.busy_time * self.cost_per_unit_time))


class FogDevice:
    # Class variable to track fog node count independently of g_devid.
    fog_node_count = 0

    # Hardware specifications for each fog node type.
    fog_node_configs = [
        (CPU_SPEED1, FD_BANDWIDTH1,
         FD_RAM1, BUSY_POWER1, IDLE_POWER1),
        (CPU_SPEED2, FD_BANDWIDTH2,
         FD_RAM2, BUSY_POWER2, IDLE_POWER2),
        (CPU_SPEED3, FD_BANDWIDTH3,
         FD_RAM3, BUSY_POWER3, IDLE_POWER3),
    ]

    @classmethod
    def reset_count(cls):
        """Reset fog node counter. Call before creating fog nodes
        to ensure correct hardware spec assignment when running
        multiple simulations in the same Python session."""
        cls.fog_node_count = 0

    def __init__(self, name):
        global g_devid
        g_devid += 1
        self.id = g_devid

        self.jobsQ = DeviceJobQ()
        self.devTime = DeviceTime()
        self.name = name + str(self.id)

        self.total_time = 0.
        self.total_Comp_time = 0.
        self.noTasksExecuted = 0

        self.slot_no_task_ex = 0
        self.slot_busy_time = 0

        # Assign hardware specs using fog node index
        fog_index = FogDevice.fog_node_count % len(FogDevice.fog_node_configs)
        if FogDevice.fog_node_count >= len(FogDevice.fog_node_configs):
            print("[WARNING] No unique specs for fog node " +
                  str(FogDevice.fog_node_count + 1) +
                  ". Reusing Fog-" + str(fog_index + 1) + " specs. " +
                  "Add new entry to fog_node_configs in devices.py.")

        config = FogDevice.fog_node_configs[fog_index]
        self.cpu_speed = config[0]
        self.bandwidth = config[1]
        self.ram = config[2]
        self.busy_power = config[3]
        self.idle_power = config[4]

        # Store the fog class index (0-based) for label mapping
        self.fog_class_index = FogDevice.fog_node_count

        FogDevice.fog_node_count += 1

        # Execution capacity in TIME units (SU).
        # This is the total available time per slot.
        self.Ex_capacity = SLOT_TIME

    def get_Ex_capacity(self):
        return self.Ex_capacity

    def get_CPUSpeed(self):
        return self.cpu_speed

    def get_bPower(self):
        return self.busy_power

    def get_iPower(self):
        return self.idle_power

    def getBANDWIDTH(self):
        return self.bandwidth

    def getRAM(self):
        return self.ram

    def get_remaining_capacity(self, sim_time):
        """Returns remaining execution capacity in TIME units (SU)
        for current slot."""
        return (sim_time + SLOT_TIME) - self.devTime.get_time()

    def clean(self, sim_time):
        self.slot_busy_time = 0
        self.slot_no_task_ex = 0
        self.devTime.set_time(sim_time)

    def ExecutesJob(self, jb, sim_time, slotno):
        """Execute a job on this fog node.

        IMPORTANT: This method no longer does horizontal offloading
        or cloud escalation. The scheduling decision (which device
        runs which task) is made entirely by the GA or AI scheduler.
        If the scheduler assigns a task here but it does not fit,
        the task is not executed and the label remains as assigned
        by the scheduler via set_ex_devid().

        This ensures features and labels in the dataset are always
        consistent: the label reflects the scheduler decision, and
        the capacity features reflect the state at decision time.
        """
        swap_time = compute_swap_time(
            jb.get_noInstructions(), jb.get_dataBytes(), self.ram)
        duration_need = compute_exec_duration(
            jb.get_noInstructions(), self.cpu_speed, swap_time)
        remaining_time = (sim_time + SLOT_TIME) - self.devTime.get_time()

        if duration_need <= remaining_time:
            self.slot_no_task_ex += 1
            jb.set_ex_devid(self.id)
            duration = jb.setExecutionDetails(
                jb.get_dataBytes() / self.bandwidth + LATENCY_ED_FR,
                swap_time,
                self.id,
                self.cpu_speed,
                slotno
            )
            self.devTime.add_time(duration)
            self.slot_busy_time += duration
        else:
            # Task does not fit. Escalate to cloud.
            # No horizontal offloading to prevent label corruption.
            cs.ExecutesJob(jb, sim_time, slotno)

    def ComputeSlotLoad(self):
        msg = "Cluster:" + self.name
        logentry.WriteStringToSlotsLog(msg)
        msg = " No Tasks Executed:" + str(self.slot_no_task_ex)
        logentry.WriteStringToSlotsLog(msg)
        self.noTasksExecuted = self.noTasksExecuted + self.slot_no_task_ex
        msg = (" CPU (MIPS):" + str(self.cpu_speed) +
               " Busy Time:" + str(self.slot_busy_time) + "\n")
        logentry.WriteStringToSlotsLog(msg)
        self.total_Comp_time = self.total_Comp_time + min(
            self.slot_busy_time, SLOT_TIME)
        self.total_time = self.total_time + SLOT_TIME
        self.devTime.set_time(self.total_time)

    def getIntervelMetricsDetails(self):
        time_reward = self.slot_busy_time * self.cpu_speed
        return time_reward

    def PrintResult(self, slot):
        total_idletime = self.total_time - self.total_Comp_time
        self.energy_utilised = (total_idletime * self.idle_power +
                                self.total_Comp_time * self.busy_power)
        TH = self.noTasksExecuted / MAX_SIMULATION_TIME
        cpu_utilization = self.total_Comp_time / self.total_time * 100
        print("Device:" + self.name)
        print("      Total Tasks Executed:" + str(self.noTasksExecuted))
        print("      Total Busy Time:" + str(self.total_Comp_time))
        print("      Total Idle Time:" + str(total_idletime))
        print("      Total Time:" + str(self.total_time))
        print("      CPU Utilization:" + str(cpu_utilization))
        print("      Total Power Utilised:" + str(self.energy_utilised))
        print("      Throughput:" + str(TH))


logentry = FileWrite()
jobsTimetracker = TimeTrackerJb()
tasksTimetracker = TimeTrackerTsk()
cs = Cloud()