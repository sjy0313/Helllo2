package ch08_interface.sec08_multi_interface;

public class SmartTelevisionExample {
	public static void main(String[] args) {
		
		RemoteControl rc = new SmartTelevision();		
		
		Searchable searchable = new SmartTelevision();
		searchable.search("https://www.youtube.com");
		
		SmartTelevision st = (SmartTelevision)rc;
		st.turnOn();
		st.search("https://www.google.com");
		st.turnOff();
	}
}
