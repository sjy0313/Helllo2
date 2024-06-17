package ch07_.inheritance.sec05_final_class.exam02_finalmethod;

public class Taxi extends Car {
	
	public void stop(boolean change) {
		if(change) {
			System.out.println("승객 하차를 위해 멈춤");
		}
			else {
				//Super.stop();
				stop();
	
			}
			
	}
}
