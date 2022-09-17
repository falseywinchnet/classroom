#trying to create an implementation for https://beautifulideas.net/unn-framework/

class School(nn.Module):
    def __init__(self):
        super(School, self).__init__()
        self.teacher = nn.ModuleList()
        self.translators = nn.ModuleList()
        self.students = nn.ModuleList()
        self.members = []
        
    def add_member(self, member):
            if member.role == "Teacher":
                self.teacher.append(member)
                self.members.append(member.idnum)
            if member.role == "Translator":
                self.translators.append(member)
                self.members.append(member.idnum)
            if member.role == "Student":
                self.students.append(member)
                self.members.append(member.idnum)
                
    def remove_member(self, idnum):
        #iterates through all school members looking for the member to remove.
        #once it has been found, it is removed from the network.
        #does this clear weights?
        for each in self.teacher:
            if self.teacher.each.id == idnum:
                delattr(self.teacher, each)
        for each in self.translators:
            if self.translators.each.id == idnum:
                delattr(self.translators, each)
        for each in self.students:
            if self.students.each.id == idnum:
                delattr(self.students, each)  
        self.members.remove(idnum)
        
                
    def evalswitch_group(self, group,eval_true):
        #enables or disables training for an entire group.
        eval_false = not eval_true
        if eval_false:
            if group == 0:
                for each in self.teacher:
                self.teacher.each.train()
            if group == 1:
                for each in self.translators:
                self.translators.each.train()
            if group == 2:
                for each in self.students:
                self.students.each.train()
        if eval_true
            if group == 0:
                for each in self.teacher:
                self.teacher.each.eval()
            if group == 1:
                for each in self.translators:
                self.translators.each.eval()
            if group == 2:
                for each in self.students:
                self.students.each.eval()

    def reset_member(self, id):
        for each in self.teacher:
            if self.teacher.each.id == idnum:
                self.teacher.each.apply(weight_reset)
        for each in self.translators:
            if self.translators.each.id == idnum:
                self.teacher.each.apply(weight_reset)
        for each in self.students:
            if self.students.each.id == idnum:
                self.teacher.each.apply(weight_reset)        
                
    def save_member(self,idnum):
        for each in self.teacher:
            if self.teacher.each.id == idnum:
                torch.save(self.teacher.each, self.teacher.each.id + '.pth')
        for each in self.translators:
            if self.translators.each.id == idnum:
                torch.save(self.translators.each, self.translators.each.id + '.pth')
        for each in self.students:
            if self.students.each.id == idnum:
                torch.save(self.students.each, self.students.each.id + '.pth') 
                
    def save_classroom(self,stage):
        classroom = nn.ModuleList()
        for each in self.teacher:
            classroom.append(self.teacher.each)
        for each in self.translators:
            classroom.append(self.translators.each)
        for each in self.students:
            classroom.append(self.students.each)
        torch.save(classroom, "stage_ " + stage + '_classroom.pth')
    
    def distill_classroom(self):
        classroom = nn.ModuleList()
        for each in self.teacher:
            classroom.append(self.teacher.each)
        for each in self.translators:
            classroom.append(self.translators.each)
        for each in self.students:
            classroom.append(self.students.each)
            
        for each in classroom[0]:
            classroom.each.optimizer.zero_grad(set_to_none=True)
            #clear .grad of the weights
            delattr(classroom.each.optimizer)
            #remove the optimizer entirely from the model for evaluation
        torch.save(classroom, 'pruned.pth')
            
        
    def empty_classroom(self):
        self.teacher = nn.ModuleList()
        self.translators = nn.ModuleList()
        self.students = nn.ModuleList()
        
    def load_classroom(self,obj):
        empty_classroom()
        for module, value in obj.__dict__.items():
            add_member(value) #no idea if this would work or not       
            
    def train(self,data, stage):
    #so what we want to do here, is we want to feed a specific input to the classroom and
    #train on that one input. We will elsewhere manually reconfigure parts of the network
    
    clean_data = data
    med = numpy.median(clean_data)
    mad =  numpy.median(numpy.abs(clean_data - med))         
    #stage 1
    if resume == False or resume_stage < 2:
    
    variants = []
    
    for each in range(25):
        noise = numpy.random.normal(0.0, mad/25, size=data.size)
        variants.append(noise + clean_data)
        
    for each in range(25):       
        m,n,o = forward(variants[each])
        loss = criterion(y1, y)
        writer.add_scalar("Loss/train", loss, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_model(10)
    writer.flush()
    
     m,n,o = forward(iter,data)
        
        

    
    #stage 2
    save_classroom(1)
    if resume == False or resume_stage == 2:
    
    #stage 3
    save_classroom(2)
    if resume == False or resume_stage == 3:
    
    #cycle 1
    
    #stage 4
    save_classroom(3)
    if resume == False or resume_stage == 4:
    
    #stage 5
    save_classroom(4)
    if resume == False or resume_stage == 5:
    
    #stage 6
    save_classroom(5)
    if resume == False or resume_stage == 6:
    
    #cycle 2

    
    #stage 7
    save_classroom(6)
    if resume == False or resume_stage == 7:
    
    #stage 8
    save_classroom(7)
    if resume == False or resume_stage == 8:
    
    #stage 9
    save_classroom(8)
    if resume == False or resume_stage == 9:
    
    #stage 10
    save_classroom(9)
    if resume == False or resume_stage == 10:
    
    #cycle 3
    save_classroom(10)
    
    
    
    distill_classroom()
    
    
    
    def forward(self, x):
        n = []
        o = []
        for each in self.teacher:
            m = self.teacher.each(x)
        for each in self.students:
            if self.students.each.parent == self.teacher[0].id:
                n.append(self.student.each(m))
            else:
            for each in self.translators:
                if self.students.each.parent == self.translators.each.id:
                    r = self.translators.each(m)
                    o.append(r)
                    n.append(self.student.each(r))
        return m, n, o
    #m = encoder output
    #n = decoder output
    #o = translator output
    


class GenericMember(nn.Module):
    def __init__(self, role, idnum, alpha, input_dims, output_dims, n_actions):
        super(GenericNetwork, self).__init__()
        self.role = role
        self.idnum = idnum
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        
    def propogate_loss(y_pred, y_true):
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(y_pred,y_true)
        self.optimizer.zero_grad()
        loss.backward()
        return loss

    def forward(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    @torch.no_grad()
    def weight_reset(m: nn.Module) -> None:
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    
    def reverse(self, x):
    #runs the module backwards with a desired output
    #must be custom-built along with forward
        return x
