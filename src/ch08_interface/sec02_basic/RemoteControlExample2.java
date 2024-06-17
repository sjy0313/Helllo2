package ch08_interface.sec02_basic;

public class RemoteControlExample2 {
	public static void main(String[] args) {
		Television tv = new Television();
		tv.turnOn();
		
		Audio audio = new Audio();
		audio.turnOn();
	}
}