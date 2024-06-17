package ch08_interface.sec02_basic;

public class RemoteControlExample3 {
	public static void main(String[] args) {
			turnOn(new Television());
			turnOn(new Audio());
	
	}
	
	static void turnOn(RemoteControl rc) {
		String who = (rc instanceof Television) ? "TV" :
					 (rc instanceof Audio) ? "Audio" : "Etc";
		System.out.printf("[%s]", who);
		rc.turnOn();
	}	
}
