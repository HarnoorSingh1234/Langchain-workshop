from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
    Space exploration is the ongoing discovery and exploration of celestial structures in outer space by means of continuously evolving and growing space technology. While the study of space is carried out mainly by astronomers with telescopes, the physical exploration of space is conducted both by unmanned robotic space probes and human spaceflight.

    These missions are conducted by various national space agencies, such as NASA, ESA, CNSA, ISRO, and private spaceflight companies such as SpaceX, Blue Origin, and Virgin Galactic. The launch of Sputnik 1 in 1957 marked the beginning of space exploration, leading to significant milestones such as the first human in space (Yuri Gagarin in 1961), the first human on the Moon (Neil Armstrong in 1969), and the establishment of the International Space Station (ISS) in 1998.
    A spy, or agent, is a person who secretly gathers information about a rival government or group to support national security. Their work often involves infiltrating enemy ranks to steal secrets, such as military strength, and can also include counterintelligence to thwart foreign espionage. This role is distinct from a professional intelligence officer and demands a life of secrecy, often requiring them to leave behind personal relationships and live with a constant awareness of danger. 
Role and duties of a spy
Information gathering: Spies collect and transmit sensitive information through various means, from modern technology to more primitive methods like invisible ink. 
Infiltration: They may infiltrate the ranks of an organization to report on troop sizes, strengths, and identify potential collaborators or dissidents. 
Counterintelligence: Spies also engage in counterintelligence to prevent foreign intelligence agencies from gathering information on their own country. 
Support for national security: Their work is crucial for a nation's safety, protecting against national threats and ensuring the success of government obligations. 
Lifestyle of a spy
Secrecy: A spy's life is defined by secrecy; they cannot reveal their true identity or work to anyone, even family and friends. 
Sacrifice: This job can require immense personal sacrifice, including leaving family and friends behind, and living a life of constant vigilance and danger. 
Skills: Spies are often intelligent, skilled individuals who may be trained in tradecraft, escape techniques, and other specialized skills to complete their missions. 

"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
)

chunks = splitter.split_text(text)

print(chunks)
# print(len(chunks))
# print(chunks[0])
# print(chunks[1])
# print(chunks[2])